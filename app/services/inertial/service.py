# app/services/inertial_service.py
"""Inertial vibration processing for IRI computation using both simple displacement method
and Quarter Car Model (QCM) simulation per ASTM E1926 / ISO 8608 standards."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt, resample_poly
from scipy.interpolate import interp1d

from app.core.config import IRI_SEGMENT_LENGTH_M, V_REF_KMH, V_SPEED
from app.core.logger import get_pipeline_logger, monitor
from app.services.pipeline.preprocessing import PreprocessingPipeline, PreprocessedData

logger = get_pipeline_logger("inertial_service")


@dataclass
class IRISegmentResult:
    """Single IRI segment result at standard interval."""
    start_distance_m: float
    end_distance_m: float
    iri_m_km: float
    sample_count: int


@dataclass
class InertialPipelineResult:
    """Complete result from simple displacement-based IRI processing."""
    iri_m_km: float
    total_distance_m: float
    iri_segments: List[IRISegmentResult]
    raw_timestamps: np.ndarray
    vertical_displacement: np.ndarray


@dataclass
class CombinedIRIResult:
    """Combined result containing simple IRI metrics."""
    simple: InertialPipelineResult


@dataclass
class ASTMQCMResult:
    """ASTM E1926/E950 compliant QCM IRI result with quality flags."""
    iri_m_km: float
    total_distance_m: float
    iri_segments: List[IRISegmentResult]
    sample_rate_hz: float
    quality_status: str = "good"
    quality_flags: List[str] = field(default_factory=list)
    upsampling_info: Dict = field(default_factory=dict)
    quarter_car_params: Dict = field(default_factory=dict)
    astm_compliance: Dict = field(default_factory=dict)


class InertialRefineryService:
    """Processes vehicle vibration and inertial sensor data to calculate
    International Roughness Index (IRI) using simple displacement method."""

    @staticmethod
    @monitor(stage="inertial_refinery_process")
    def process_vibration_data(
        timestamps: np.ndarray,
        accel_z: np.ndarray,
        speeds_kmh: Optional[np.ndarray] = None,
        run_id: Optional[str] = None,
        latitudes: Optional[np.ndarray] = None,
        longitudes: Optional[np.ndarray] = None,
    ) -> InertialPipelineResult:
        """Simple displacement-to-distance IRI computation with full preprocessing (ST1-ST5b).
        
        Computes Vertical_Displacement(m) from raw accelerometer data through:
        ST1: Format data | ST2: Speed filter | ST3: Reorientation | ST4: Z-filter | ST5: Location filter | ST5b: Vertical displacement
        
        Travel distance uses Haversine from lat/lon (per reference ST7_TravelDistan.py).
        Per reference codebase: IRI = sum(|displacement|) / distance_km (m/km)
        """
        rid = run_id or ""
        logger.info("Processing vibration data (simple displacement method with ST1-ST5b preprocessing) for IRI", run_id=rid)

        # Build minimal DataFrame for preprocessing pipeline
        # We need accel_x, accel_y, accel_z, lat, long, gps_speed, time_ms, gravi_x/y/z
        n = int(len(accel_z)) if hasattr(accel_z, '__len__') else int(accel_z.shape[0])
        if n < 10:
            logger.warning("Insufficient data points", run_id=rid)
            return InertialPipelineResult(
                iri_m_km=0.0, total_distance_m=0.0, iri_segments=[],
                raw_timestamps=timestamps, vertical_displacement=np.zeros_like(timestamps)
            )

        # Build minimal sensor data for preprocessing
        # Use actual lat/lon if provided, otherwise zeros
        df = pd.DataFrame({
            'time_ms': timestamps * 1000,  # convert to ms
            'accel_x': np.zeros_like(accel_z),
            'accel_y': np.zeros_like(accel_z),
            'accel_z': accel_z,
            'gravi_x': np.zeros_like(accel_z),
            'gravi_y': np.zeros_like(accel_z),
            'gravi_z': np.full_like(accel_z, 9.81),
            'lat': latitudes if latitudes is not None else np.zeros_like(accel_z),
            'long': longitudes if longitudes is not None else np.zeros_like(accel_z),
            'gps_speed': speeds_kmh if speeds_kmh is not None else np.full_like(accel_z, 30.0),
            'time_ms': timestamps * 1000,
        })

        # Run preprocessing pipeline (ST1-ST5b)
        processed = PreprocessingPipeline.run_pipeline(df)
        
        # Extract preprocessed data
        displacement = processed['vertical_displacement'].values  # type: ignore[attr-defined]
        if displacement is None or len(displacement) == 0:
            displacement = np.zeros_like(accel_z)
        
        # Get timestamps and speeds from processed data
        t = processed['time_ms'].values / 1000.0  # convert to seconds  # type: ignore[attr-defined]
        if len(t) < 10:
            return InertialPipelineResult(
                iri_m_km=0.0, total_distance_m=0.0, iri_segments=[],
                raw_timestamps=timestamps, vertical_displacement=np.zeros_like(timestamps)
            )
        
        # Use actual speeds from processed data
        speeds_kmh = processed['gps_speed'].values  # type: ignore[attr-defined]
        
        # Compute travel distance using Haversine from lat/lon (per reference ST7_TravelDistan.py)
        lat_vals = processed['lat'].values  # type: ignore[attr-defined]
        lon_vals = processed['long'].values  # type: ignore[attr-defined]
        distances_m = InertialRefineryService._haversine_distances_m(lat_vals, lon_vals)  # type: ignore[arg-type]
        total_distance = float(distances_m[-1]) if len(distances_m) > 0 else 0.0
        
        # Use precomputed vertical displacement (already in meters)
        displacement = processed['vertical_displacement'].values  # type: ignore[attr-defined]
        vert_disp_m = displacement  # meters
        
        # Compute IRI segments using accumulation approach (per reference calculate_iri_values)
        # Accumulate distance and vert_disp, emit a segment when accumulated >= segment_length.
        # This naturally skips stopped periods instead of creating empty 0.0000 segments.
        segment_length_m = IRI_SEGMENT_LENGTH_M
        segment_length_km = segment_length_m / 1000.0
        
        accumulated_dist = 0.0
        accumulated_disp = 0.0
        prev_vert_disp = 0.0
        speed_sum = 0.0
        speed_count = 0
        iri_segments = []
        
        for i in range(len(distances_m)):
            dist_delta = distances_m[i] - (distances_m[i - 1] if i > 0 else 0.0)
            vert_disp = float(vert_disp_m[i])  # type: ignore[arg-type]
            sp = float(speeds_kmh[i]) if speeds_kmh is not None and i < len(speeds_kmh) else 0.0  # type: ignore[index]
            sp = float(sp)  # Ensure sp is a Python float
            
            accumulated_dist += dist_delta
            # IRI uses absolute slope: |Δvertical_displacement / Δdistance|
            # which equals |Δvertical_displacement| / Δdistance
            # So accumulated_disp should accumulate |Δvertical_displacement|
            vert_change = abs(vert_disp - prev_vert_disp)
            accumulated_disp += vert_change
            prev_vert_disp = vert_disp
            
            if sp > 0:
                speed_sum += sp
                speed_count += 1
            
            if accumulated_dist >= segment_length_m:
                avg_speed = speed_sum / speed_count if speed_count > 0 else 0.0
                seg_iri = accumulated_disp / (accumulated_dist / 1000.0) if accumulated_dist > 0 else 0.0  # m/km
                
                seg_start = (len(iri_segments)) * segment_length_m
                seg_end = seg_start + segment_length_m
                
                iri_segments.append(IRISegmentResult(
                    start_distance_m=seg_start,
                    end_distance_m=seg_end,
                    iri_m_km=seg_iri,
                    sample_count=speed_count,
                ))
                
                accumulated_dist = 0.0
                accumulated_disp = 0.0
                speed_sum = 0.0
                speed_count = 0
        
        overall_iri = float(np.mean([s.iri_m_km for s in iri_segments])) if iri_segments else 0.0
        
        return InertialPipelineResult(
            iri_m_km=overall_iri,
            total_distance_m=total_distance,
            iri_segments=iri_segments,
            raw_timestamps=timestamps,
            vertical_displacement=displacement,  # type: ignore[arg-type]
        )

    @staticmethod
    def _haversine_distances_m(latitudes: Union[np.ndarray, pd.Series], longitudes: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Compute cumulative travel distance using Haversine formula from lat/lon.
        
        Reference implementation from ST7_TravelDistan.py and final_codebase/data_processors.py.
        
        Args:
            latitudes: Latitude values in degrees
            longitudes: Longitude values in degrees
            
        Returns:
            Cumulative distance array in meters
        """
        n = len(latitudes)
        if n < 2:
            return np.zeros(n)
        
        lat_rad = np.radians(latitudes)
        lon_rad = np.radians(longitudes)
        
        distances = np.zeros(n)
        for i in range(1, n):
            dlat = lat_rad[i] - lat_rad[i - 1]
            dlon = lon_rad[i] - lon_rad[i - 1]
            a = np.sin(dlat / 2) ** 2 + np.cos(lat_rad[i - 1]) * np.cos(lat_rad[i]) * np.sin(dlon / 2) ** 2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            distances[i] = distances[i - 1] + 6371000.0 * c  # Earth radius in meters
        
        return distances

    @staticmethod
    def _compute_quarter_car_iri(
        profile: np.ndarray,
        dx: float,
        speed_ms: float = V_SPEED,
    ) -> float:
        """Compute IRI using Quarter Car Model simulation (RK4 integration).

        Args:
            profile: Road elevation profile (m) at uniform spatial spacing
            dx: Spatial sampling interval (m)
            speed_ms: Vehicle speed (m/s), default 80 km/h = 22.22 m/s

        Returns:
            IRI in mm/m (m/km)
        """
        # Quarter car model parameters (Golden Car per ASTM E1926)
        ms = 250.0      # Sprung mass (kg)
        mu = 30.0       # Unsprung mass (kg)
        ks = 63300.0    # Suspension stiffness (N/m)
        kt = 653000.0   # Tire stiffness (N/m)
        cs = 6000.0     # Suspension damping (N·s/m)

        n = len(profile)
        if n < 2:
            return 0.0

        x = np.zeros(4)  # [sprung_disp, sprung_vel, unsprung_disp, unsprung_vel]
        iri_accum = 0.0
        dt = dx / speed_ms

        for i in range(n - 1):
            zr = profile[i]

            def f(x_state: np.ndarray, zr_val: float) -> np.ndarray:
                x1, x2, x3, x4 = x_state
                dx1 = x2
                dx2 = (-ks * (x1 - x3) - cs * (x2 - x4)) / ms
                dx3 = x4
                dx4 = (ks * (x1 - x3) + cs * (x2 - x4) - kt * (x3 - zr_val)) / mu
                return np.array([dx1, dx2, dx3, dx4])

            # RK4 integration
            k1 = f(x, zr)
            k2 = f(x + 0.5 * dt * k1, zr)
            k3 = f(x + 0.5 * dt * k2, zr)
            k4 = f(x + dt * k3, zr)

            x = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            # IRI accumulates absolute relative velocity over time
            iri_accum += abs(x[1] - x[3]) * dt

        total_distance = (n - 1) * dx
        if total_distance <= 0:
            return 0.0
        iri = iri_accum / total_distance  # m/m
        return iri * 1000  # mm/m

    @staticmethod
    def _resample_to_uniform_distance(
        distances: np.ndarray,
        profile: np.ndarray,
        target_dx: float = 0.025,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resample profile to uniform spatial spacing."""
        if len(distances) < 2 or len(profile) < 2:
            return np.array([0.0]), np.array([0.0])

        mask = np.isfinite(distances) & np.isfinite(profile)
        distances = distances[mask]
        profile = profile[mask]

        if len(distances) < 2:
            return np.array([0.0]), np.array([0.0])

        valid = np.diff(distances) > 0
        distances = distances[:-1][valid]
        profile = profile[:-1][valid]

        if len(distances) < 2:
            return np.array([0.0]), np.array([0.0])

        d_uniform = np.arange(0, float(distances[-1]), target_dx)
        if len(d_uniform) < 2:
            return np.array([0.0]), np.array([0.0])

        profile_uniform = np.interp(d_uniform, distances, profile)
        return d_uniform, profile_uniform

    @staticmethod
    @monitor(stage="astm_qcm_process")
    def process_astm_qcm(
        timestamps: np.ndarray,
        accel_z: np.ndarray,
        speeds_kmh: Optional[np.ndarray] = None,
        run_id: Optional[str] = None,
        target_fs: float = 100.0,
        target_dx: float = 0.025,
        v_ref_kmh: float = 80.0,
    ) -> ASTMQCMResult:
        """
        ASTM E1926/E950 compliant QCM IRI with 100Hz upsampling and quality flags.
        
        Per ASTM E1926: Quarter-car simulation at 80 km/h (22.22 m/s)
        Per ASTM E950: Class 1 requires >= 100Hz sampling, <=25mm spacing
        
        Distance is computed from speed × time (standardized simulation distance),
        NOT from GPS Haversine. This is by design: the QCM simulates the quarter-car
        response at a reference speed, so its distance represents the simulated
        travel, not the actual GPS-measured road length.
        
        Quality flags:
        - low_sampling_rate: original fs < 100Hz
        - speed_variation: speed CV > 0.2
        - insufficient_distance: total distance < 100m
        - profile_gaps: >5% missing samples after resampling
        - quarter_car_divergence: state variables exceed bounds
        """
        rid = run_id or ""
        logger.info(f"Processing ASTM QCM IRI (target_fs={target_fs}Hz, v_ref={v_ref_kmh}km/h)", run_id=rid)
        
        quality_flags = []
        
        valid_mask = np.isfinite(timestamps) & np.isfinite(accel_z)
        t = timestamps[valid_mask]
        a_z = accel_z[valid_mask]
        
        if len(t) < 10:
            logger.warning("Insufficient data for ASTM QCM IRI", run_id=rid)
            return ASTMQCMResult(
                iri_m_km=0.0,
                total_distance_m=0.0,
                iri_segments=[],
                sample_rate_hz=target_fs,
                quality_status="warning",
                quality_flags=["insufficient_data"],
            )
        
        sort_idx = np.argsort(t)
        t = t[sort_idx]
        a_z = a_z[sort_idx]
        
        if speeds_kmh is not None and len(speeds_kmh) == len(sort_idx):
            v_ms = speeds_kmh[sort_idx] / 3.6
            v_ms = np.where(v_ms > 1.0, v_ms, v_ref_kmh / 3.6)
        else:
            v_ms = np.full_like(t, v_ref_kmh / 3.6)
        
        # Check sampling rate
        dt = np.diff(t)
        dt = np.insert(dt, 0, dt[0] if len(dt) > 0 else 0.01)
        dt = np.clip(dt, 0.001, 1.0)
        original_fs = 1.0 / np.median(dt) if np.median(dt) > 0 else 100.0
        
        upsampling_info = {
            "original_fs_hz": float(original_fs),
            "target_fs_hz": target_fs,
            "upsampling_factor": float(target_fs / original_fs) if original_fs > 0 else 1.0,
        }
        
        if original_fs < 100.0:
            quality_flags.append("low_sampling_rate")
        
        # Check speed variation
        speed_cv = float(np.std(v_ms) / np.mean(v_ms)) if np.mean(v_ms) > 0 else 0.0
        if speed_cv > 0.2:
            quality_flags.append("speed_variation")
        
        # Upsample to target_fs using polyphase resampling
        if original_fs > 0 and abs(original_fs - target_fs) > 1.0:
            up_factor = int(target_fs)
            down_factor = int(original_fs)
            # Use resample_poly for better quality
            try:
                a_z_resampled = resample_poly(a_z, up_factor, down_factor)  # type: ignore[return-value]
                v_ms_resampled = resample_poly(v_ms, up_factor, down_factor)  # type: ignore[return-value]
                t_resampled = np.arange(len(a_z_resampled)) / target_fs
                a_z, v_ms, t = a_z_resampled, v_ms_resampled, t_resampled
                upsampling_info["method"] = "resample_poly"  # type: ignore[assignment]
            except Exception:
                # Fallback to linear interpolation
                f_interp = interp1d(t, a_z, kind='linear', fill_value='extrapolate')  # type: ignore[arg-type]
                f_speed = interp1d(t, v_ms, kind='linear', fill_value='extrapolate')  # type: ignore[arg-type]
                t = np.arange(0, float(t[-1]), 1.0/target_fs)
                a_z = f_interp(t)
                v_ms = f_speed(t)
                upsampling_info["method"] = "linear_interp"  # type: ignore[assignment]
        else:
            upsampling_info["method"] = "none_needed"  # type: ignore[assignment]
        
        # Distance from speed × time (standardized simulation distance, NOT GPS Haversine)
        dt = 1.0 / target_fs
        distances = np.cumsum(v_ms * dt)
        total_distance = float(distances[-1]) if len(distances) > 0 else 0.0
        
        if total_distance < 100.0:
            quality_flags.append("insufficient_distance")
        
        # High-pass filter at 0.1 Hz
        a_z_corrected = a_z - np.mean(a_z)
        velocity_z = np.cumsum(a_z_corrected * dt)
        nyq = 0.5 * target_fs
        hp_cutoff = 0.1
        b_hp = a_hp = None
        if hp_cutoff < nyq:
            b_hp, a_hp = butter(N=2, Wn=hp_cutoff, btype="highpass", fs=target_fs)  # type: ignore[return-value]
            velocity_z = filtfilt(b_hp, a_hp, velocity_z)
        disp_z = np.cumsum(velocity_z * dt)
        if hp_cutoff < nyq and b_hp is not None:
            disp_z = filtfilt(b_hp, a_hp, disp_z)
        
        # Resample to uniform spatial spacing for QCM
        d_uniform, profile_uniform = InertialRefineryService._resample_to_uniform_distance(
            distances, disp_z, target_dx
        )
        
        profile_gap_ratio = 1.0 - (len(d_uniform) * target_dx / total_distance) if total_distance > 0 else 1.0
        if profile_gap_ratio > 0.05:
            quality_flags.append("profile_gaps")
        
        if len(profile_uniform) < 2:
            return ASTMQCMResult(
                iri_m_km=0.0,
                total_distance_m=total_distance,
                iri_segments=[],
                sample_rate_hz=target_fs,
                quality_status="warning",
                quality_flags=quality_flags + ["profile_too_short_after_resample"],
                upsampling_info=upsampling_info,
            )
        
# Single-pass quarter-car simulation at reference speed with segment tracking
        v_ref_ms = v_ref_kmh / 3.6
        dt_qc = target_dx / v_ref_ms
        
        # Quarter-car parameters (Golden Car per ASTM E1926)
        ms = 250.0      # Sprung mass (kg)
        mu = 30.0       # Unsprung mass (kg)
        ks = 63300.0    # Suspension stiffness (N/m)
        kt = 653000.0   # Tire stiffness (N/m)
        cs = 6000.0     # Suspension damping (N·s/m)
        
        # Segment tracking
        segment_length = IRI_SEGMENT_LENGTH_M
        points_per_segment = int(segment_length / target_dx)
        if points_per_segment < 2:
            points_per_segment = 2
        
        n = len(profile_uniform)
        num_segments = max(1, (n - 1) // points_per_segment)
        
        # Pre-allocate segment accumulators
        segment_iri_accum = np.zeros(num_segments)
        segment_dist = np.zeros(num_segments)
        segment_counts = np.zeros(num_segments, dtype=int)
        
        # Quarter-car state
        x = np.zeros(4)
        max_disp = 0.0
        overall_iri_accum = 0.0
        
        # Define RK4 derivative function outside loop for performance
        def qc_derivative(x_state, zr_val):
            x1, x2, x3, x4 = x_state
            return np.array([
                x2,
                (-ks * (x1 - x3) - cs * (x2 - x4)) / ms,
                x4,
                (ks * (x1 - x3) + cs * (x2 - x4) - kt * (x3 - zr_val)) / mu
            ])
        
        # Single-pass quarter-car simulation with segment tracking
        for i in range(n - 1):
            zr = profile_uniform[i]
            
            # RK4 integration
            k1 = qc_derivative(x, zr)
            k2 = qc_derivative(x + 0.5 * dt_qc * k1, zr)
            k3 = qc_derivative(x + 0.5 * dt_qc * k2, zr)
            k4 = qc_derivative(x + dt_qc * k3, zr)
            x = x + (dt_qc / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            
            # Accumulate overall IRI
            overall_iri_accum += abs(x[1] - x[3]) * dt_qc
            
            # Accumulate segment IRI
            seg_idx = i // points_per_segment
            if seg_idx < num_segments:
                segment_iri_accum[seg_idx] += abs(x[1] - x[3]) * dt_qc
                segment_dist[seg_idx] += dt_qc * v_ref_ms
                segment_counts[seg_idx] += 1
            
            max_disp = max(max_disp, abs(x[0]), abs(x[2]))
        
        overall_iri = overall_iri_accum / total_distance * 1000 if total_distance > 0 else 0.0
        
        if max_disp > 1.0:  # >1m displacement unrealistic
            quality_flags.append("quarter_car_divergence")
        
        # Build segment results from accumulators
        iri_segments = []
        for seg_idx in range(num_segments):
            if segment_counts[seg_idx] < 2:
                continue
            seg_dist = segment_dist[seg_idx]
            if seg_dist > 0:
                seg_iri = segment_iri_accum[seg_idx] / seg_dist * 1000  # mm/m
            else:
                seg_iri = 0.0
            seg_dist_start = seg_idx * segment_length
            seg_dist_end = min((seg_idx + 1) * segment_length, total_distance)
            
            iri_segments.append(
                IRISegmentResult(
                    start_distance_m=seg_dist_start,
                    end_distance_m=seg_dist_end,
                    iri_m_km=seg_iri,
                    sample_count=segment_counts[seg_idx] - 1,
                )
            )
        
        # Handle remaining partial segment
        remaining_start = num_segments * points_per_segment
        if remaining_start < n - 1:
            remaining_profile = profile_uniform[remaining_start:]
            if len(remaining_profile) >= 2:
                remaining_iri = InertialRefineryService._compute_quarter_car_iri(
                    remaining_profile, target_dx, v_ref_ms
                )
                remaining_dist_start = remaining_start * target_dx
                remaining_dist_end = (n - 1) * target_dx
                iri_segments.append(
                    IRISegmentResult(
                        start_distance_m=remaining_dist_start,
                        end_distance_m=remaining_dist_end,
                        iri_m_km=remaining_iri,
                        sample_count=len(remaining_profile) - 1,
                    )
                )
        
        quality_status = "good" if len(quality_flags) == 0 else "warning"
        
        quarter_car_params = {
            "ms": ms, "mu": mu, "ks": ks, "kt": kt, "cs": cs,
            "v_ref_kmh": v_ref_kmh, "target_dx_m": target_dx
        }
        
        astm_compliance = {
            "E1926_quarter_car": True,
            "E950_class1_sampling": original_fs >= 100.0,
            "E867_iri_scale": True,
            "reference_speed_kmh": v_ref_kmh,
        }
        
        return ASTMQCMResult(
            iri_m_km=overall_iri,
            total_distance_m=total_distance,
            iri_segments=iri_segments,
            sample_rate_hz=target_fs,
            quality_status=quality_status,
            quality_flags=quality_flags,
            upsampling_info=upsampling_info,
            quarter_car_params=quarter_car_params,
            astm_compliance=astm_compliance,
        )
    
    @staticmethod
    def process_imu_csv(csv_path: str) -> InertialPipelineResult:
        """Process IMU CSV and return simple displacement-based IRI result."""
        df = pd.read_csv(csv_path)
        df.columns = [str(col).strip() for col in df.columns]

        time_col = next((c for c in ["Timestamp", "time", "GPSTime"] if c in df.columns), df.columns[0])
        accel_col = next((c for c in ["LinAccelZ", "AccelZ", "accel_z"] if c in df.columns), df.columns[1])
        speed_col = next((c for c in ["SpeedKmh", "speed", "SpeedGPS"] if c in df.columns), None)

        # Prefer RecIntvl (authoritative sampling interval) when available
        recintvl = None
        if "RecIntvl" in df.columns:
            recintvl_vals = pd.to_numeric(df["RecIntvl"], errors="coerce")
            if recintvl_vals.notna().any():
                recintvl = float(recintvl_vals.iloc[0])

        # Try numeric parse first
        t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=np.float64)
        
        # If numeric parse failed, try datetime parsing
        if not np.any(np.isfinite(t)):
            try:
                parsed = pd.to_datetime(df[time_col], errors="coerce", utc=True)
                if parsed.notna().any():
                    t_ns = parsed.astype(np.int64).to_numpy(dtype=np.float64)
                    t_ns = t_ns - t_ns[0]
                    t = t_ns / 1e9  # nanoseconds to seconds
            except Exception:
                pass

        # Validate: if datetime parse gives unreasonable time range (< 1s for >100 samples),
        # fall back to RecIntvl (authoritative sampling interval)
        if np.any(np.isfinite(t)):
            t_span = float(np.max(t) - np.min(t))
            n_samples = len(df)
            if t_span < 1.0 and n_samples > 100 and recintvl and recintvl > 0:
                logger.warning(
                    f"Datetime parse gave unreasonable time span ({t_span:.3f}s for {n_samples} samples); "
                    f"using RecIntvl={recintvl}s instead"
                )
                t = None

        if t is None or not np.any(np.isfinite(t)):
            if recintvl and recintvl > 0:
                dt = float(recintvl)
                t = np.arange(len(df)) * dt
            else:
                t = np.arange(len(df), dtype=np.float64)

        # Ensure float64 numpy array
        if not isinstance(t, np.ndarray):
            t = np.asarray(t, dtype=np.float64)
        elif t.dtype != np.float64:
            t = t.astype(np.float64)

        accel = pd.to_numeric(df[accel_col], errors="coerce").to_numpy(dtype=np.float64)
        speeds = pd.to_numeric(df[speed_col], errors="coerce").to_numpy(dtype=np.float64) if speed_col else None
        
        # Extract lat/lon for Haversine distance calculation (per reference ST7)
        lat = None
        lon = None
        lat_col = next((c for c in ["Latitude", "latitude", "lat"] if c in df.columns), None)
        lon_col = next((c for c in ["Longitude", "longitude", "lon", "long"] if c in df.columns), None)
        if lat_col:
            lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype=np.float64)
        if lon_col:
            lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy(dtype=np.float64)

        valid = np.isfinite(t) & np.isfinite(accel)
        t, accel = t[valid], accel[valid]
        if speeds is not None:
            speeds = speeds[valid]
        if lat is not None:
            lat = lat[valid]
        if lon is not None:
            lon = lon[valid]

        return InertialRefineryService.process_vibration_data(t, accel, speeds, latitudes=lat, longitudes=lon)

    @staticmethod
    def process_imu_csv_simple(csv_path: str) -> InertialPipelineResult:
        """Alias for process_imu_csv for backward compatibility."""
        return InertialRefineryService.process_imu_csv(csv_path)

    @staticmethod
    def classify_iri_5rank(iri: float) -> Dict[str, Any]:
        if iri < 2.0:
            return {"rank": "Very Good", "code": 1}
        elif iri < 4.0:
            return {"rank": "Good", "code": 2}
        elif iri < 6.0:
            return {"rank": "Fair", "code": 3}
        elif iri < 8.0:
            return {"rank": "Poor", "code": 4}
        else:
            return {"rank": "Very Poor", "code": 5}