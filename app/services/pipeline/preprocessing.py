# app/services/preprocessing_pipeline.py
"""Preprocessing pipeline for IRI computation (ST1-ST5b from reference codebase).
Computes Vertical_Displacement(m) from raw accelerometer data through:
ST1: Format data (rename/reorder columns)
ST2: Speed filtering (remove stationary & GPS speed < 5 km/h)
ST3: Reorientation using gravity vector (quaternion rotation)
ST4: Z-axis filtering (Butterworth low-pass 0.1 Hz, fs=1 Hz, order=4)
ST5: Location filtering (remove GPS duplicates)
ST5b: Calculate vertical displacement (detrend + filter + integrate + clip)
"""
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd

from app.core.config import V_SPEED
from app.core.logger import get_pipeline_logger, monitor

logger = get_pipeline_logger("preprocessing_pipeline")


@dataclass
class PreprocessedData:
    """Result of preprocessing pipeline."""
    timestamps: np.ndarray          # seconds from start
    accel_x: np.ndarray             # reoriented X acceleration (m/s²)
    accel_y: np.ndarray             # reoriented Y acceleration (m/s²)
    accel_z: np.ndarray             # reoriented Z acceleration (m/s²) - filtered
    speeds_kmh: np.ndarray          # GPS speeds (km/h)
    latitudes: np.ndarray           # snapped latitudes
    longitudes: np.ndarray          # snapped longitudes
    timestamps_ms: np.ndarray       # original timestamps (ms)
    vertical_displacement: np.ndarray  # computed vertical displacement (m)
    distances_m: np.ndarray         # cumulative travel distance (m)
    total_distance_m: float         # total travel distance (m)
    sample_rate_hz: float           # effective sampling rate


class PreprocessingPipeline:
    """Preprocessing pipeline for IRI computation (ST1-ST5b)."""

    # Filter parameters from reference
    LP_CUTOFF_HZ = 20.0     # Low-pass cutoff (Hz) - raised from 0.1 to 20 Hz to preserve road profile (0.1-30 Hz spatial)
    LP_ORDER = 4            # Filter order
    FS_HZ = 100.0           # Sampling frequency (Hz) - typical for phone sensors after upsampling
    SPEED_THRESHOLD_KMH = 5.0  # Minimum GPS speed (km/h)
    STATIONARY_THRESHOLD = 0.1  # Stationary threshold (g)
    MAX_DISPLACEMENT_M = 2.0    # Max vertical displacement clip (m) — thesis treats >2 m as an outlier
    MAX_SPEED_KMH = 100.0       # Max reasonable GPS speed (km/h)

    @staticmethod
    @monitor(stage="preprocessing_st1_format")
    def st1_format_data(df: pd.DataFrame) -> pd.DataFrame:
        """ST1: Format data - rename/reorder columns to standard format."""
        logger.info("ST1: Formatting data columns")
        
        # Column mapping from phone sensor format to standard
        column_mapping = {
            'ACCELEROMETER X (m/s²)': 'accel_x',
            'ACCELEROMETER Y (m/s²)': 'accel_y',
            'ACCELEROMETER Z (m/s²)': 'accel_z',
            'GRAVITY X (m/s²)': 'gravi_x',
            'GRAVITY Y (m/s²)': 'gravi_y',
            'GRAVITY Z (m/s²)': 'gravi_z',
            'LINEAR ACCELERATION X (m/s²)': 'accel_x',
            'LINEAR ACCELERATION Y (m/s²)': 'accel_y',
            'LINEAR ACCELERATION Z (m/s²)': 'accel_z',
            'GYROSCOPE X (°/s)': 'gyros_x',
            'GYROSCOPE Y (°/s)': 'gyros_y',
            'GYROSCOPE Z (°/s)': 'gyros_z',
            'LOCATION Latitude : ': 'lat',
            'LOCATION Longitude : ': 'long',
            'LOCATION Speed ( Kmh)': 'gps_speed',
            'LOCATION Accuracy ( m)': 'Loc_Accuarcy',
            'Time since start in ms ': 'time_ms',
            'YYYY-MO-DD HH-MI-SS_SSS': 'date',
        }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required = ['accel_x', 'accel_y', 'accel_z', 'lat', 'long', 'gps_speed', 'time_ms']
        for col in required:
            if col not in df.columns:
                logger.warning(f"Missing required column: {col}")
        
        # Convert numeric columns
        numeric_cols = ['accel_x', 'accel_y', 'accel_z', 'lat', 'long', 'gps_speed', 'time_ms']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"ST1: Formatted {len(df)} rows")
        return df

    @staticmethod
    @monitor(stage="preprocessing_st2_speed_filter")
    def st2_speed_filter(df: pd.DataFrame) -> pd.DataFrame:
        """ST2: Speed filtering - remove stationary & GPS speed < 5 km/h."""
        logger.info("ST2: Applying speed filter")
        
        initial_len = len(df)
        
        # Remove stationary (acceleration magnitude < threshold)
        if all(c in df.columns for c in ['accel_x', 'accel_y', 'accel_z']):
            accel_x = df['accel_x'].to_numpy()
            accel_y = df['accel_y'].to_numpy()
            accel_z = df['accel_z'].to_numpy()
            accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
            df = df[accel_mag >= PreprocessingPipeline.STATIONARY_THRESHOLD]
        
        # Remove GPS speed below threshold
        if 'gps_speed' in df.columns:
            df = df[df['gps_speed'] >= PreprocessingPipeline.SPEED_THRESHOLD_KMH]
            df = df[df['gps_speed'] <= PreprocessingPipeline.MAX_SPEED_KMH]
        
        logger.info(f"ST2: Filtered {initial_len - len(df)} rows ({len(df)} remaining)")
        return df

    @staticmethod
    @monitor(stage="preprocessing_st3_reorient")
    def st3_reorient(df: pd.DataFrame) -> pd.DataFrame:
        """ST3: Reorientation using gravity vector (quaternion rotation).
        
        Reorients accelerometer from phone frame to vehicle frame using gravity vector.
        """
        logger.info("ST3: Reorienting accelerometer data")
        
        if not all(c in df.columns for c in ['accel_x', 'accel_y', 'accel_z', 'gravi_x', 'gravi_y', 'gravi_z']):
            logger.warning("Gravity vector columns missing, skipping reorientation")
            return df
        
        # Use gravity vector to compute orientation
        gx = df['gravi_x'].to_numpy()
        gy = df['gravi_y'].to_numpy()
        gz = df['gravi_z'].to_numpy()
        ax = df['accel_x'].to_numpy()
        ay = df['accel_y'].to_numpy()
        az = df['accel_z'].to_numpy()
        
        # Normalize gravity vector
        g_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        g_mag = np.where(g_mag > 0, g_mag, 1.0)
        gx_n, gy_n, gz_n = gx / g_mag, gy / g_mag, gz / g_mag
        
        # Reorient: project acceleration onto gravity-aligned frame
        # Z' = a · ĝ (vertical component along gravity)
        # Horizontal components in plane perpendicular to gravity
        a_dot_g = ax * gx_n + ay * gy_n + az * gz_n
        
        # Reoriented Z (vertical along gravity)
        az_reoriented = a_dot_g
        
        # Reoriented X and Y (horizontal plane)
        # Project onto plane perpendicular to gravity
        ax_h = ax - a_dot_g * gx_n
        ay_h = ay - a_dot_g * gy_n
        
        # Store reoriented accelerations
        df['accel_x'] = ax_h
        df['accel_y'] = ay_h
        df['accel_z'] = az_reoriented
        
        logger.info("ST3: Reorientation complete")
        return df

    @staticmethod
    @monitor(stage="preprocessing_st4_filter_z")
    def st4_filter_z(df: pd.DataFrame) -> pd.DataFrame:
        """ST4: Z-axis - no filtering here. 
        
        The correct IRI filter strategy (per ASTM E1926 and QCM reference):
        1. Remove mean from acceleration
        2. Integrate to velocity
        3. High-pass filter velocity at 0.1 Hz
        4. Integrate to displacement
        5. High-pass filter displacement at 0.1 Hz
        
        This is done in ST5b, not here. ST4 just passes through.
        """
        logger.info("ST4: Passing through acceleration (filtering done in ST5b)")
        return df

    @staticmethod
    @monitor(stage="preprocessing_st5_location_filter")
    def st5_location_filter(df: pd.DataFrame) -> pd.DataFrame:
        """ST5: Location filtering - remove GPS duplicates by lat/lon.
        
        Only applies if we have meaningful GPS coordinates (not all zeros).
        """
        logger.info("ST5: Filtering GPS duplicates")
        
        initial_len = len(df)
        
        if 'lat' in df.columns and 'long' in df.columns:
            # Check if we have meaningful GPS variation
            lat_std = df['lat'].std()
            long_std = df['long'].std()
            
            if lat_std > 1e-6 or long_std > 1e-6:
                # We have meaningful GPS variation, remove duplicates
                df = df.drop_duplicates(subset=['lat', 'long'], keep='first')
                df = df.reset_index(drop=True)
                logger.info(f"ST5: Removed {initial_len - len(df)} GPS duplicates")
            else:
                logger.info("ST5: No meaningful GPS variation, skipping duplicate removal")
        
        return df

    @staticmethod
    @monitor(stage="preprocessing_st5b_vertical_disp")
    def st5b_vertical_displacement(df: pd.DataFrame) -> pd.DataFrame:
        """ST5b: Calculate vertical displacement (Enochs thesis simple method).

        Per the thesis (L348-351): integrate acceleration -> velocity ->
        displacement with the trapezoidal rule, then treat |displacement|
        beyond MAX_DISPLACEMENT_M as outliers. The thesis uses the simple
        displacement method, NOT the ASTM E1926 quarter-car filter sequence,
        so no high-pass filtering or detrending is applied here: those steps
        oscillate the displacement and inflate the per-segment IRI.
        """
        logger.info("ST5b: Computing vertical displacement (thesis simple method)")

        if 'accel_z' not in df.columns or 'time_ms' not in df.columns:
            logger.warning("Required columns missing for displacement calculation")
            df['vertical_displacement'] = 0.0
            return df

        accel = df['accel_z'].to_numpy()
        n = len(accel)

        if n < 3:
            df['vertical_displacement'] = 0.0
            return df

        # 1. Remove mean from acceleration (removes gravity/static bias)
        accel_corrected = accel - np.mean(accel)

        # Time intervals in seconds
        dt = np.diff(df['time_ms'].to_numpy() / 1000.0)
        dt = np.insert(dt, 0, dt[0] if len(dt) > 0 else 0.01)
        dt = np.clip(dt, 0.001, 1.0)

        # 2. Integrate acceleration -> velocity (trapezoidal rule)
        velocity = np.zeros(n)
        for i in range(1, n):
            velocity[i] = velocity[i-1] + 0.5 * (accel_corrected[i] + accel_corrected[i-1]) * dt[i]

        # 3. Integrate velocity -> displacement (double trapezoidal rule)
        displacement = np.zeros(n)
        for i in range(1, n):
            displacement[i] = displacement[i-1] + 0.5 * (velocity[i] + velocity[i-1]) * dt[i]

        # 4. Clip |displacement| > MAX_DISPLACEMENT_M (thesis: outliers)
        max_disp = PreprocessingPipeline.MAX_DISPLACEMENT_M
        displacement = np.clip(displacement, -max_disp, max_disp)

        df['vertical_displacement'] = displacement
        logger.info(f"ST5b: Vertical displacement computed (range: {displacement.min():.6f} to {displacement.max():.6f} m)")

        return df

    @classmethod
    @monitor(stage="preprocessing_pipeline")
    def run_pipeline(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Run complete preprocessing pipeline (ST1-ST5b)."""
        logger.info("Starting preprocessing pipeline (ST1-ST5b)")
        
        df = cls.st1_format_data(df)
        df = cls.st2_speed_filter(df)
        df = cls.st3_reorient(df)
        df = cls.st4_filter_z(df)
        df = cls.st5_location_filter(df)
        df = cls.st5b_vertical_displacement(df)
        
        logger.info(f"Preprocessing pipeline complete: {len(df)} rows")
        return df

    @classmethod
    def compute_vertical_displacement_from_raw(
        cls,
        timestamps: np.ndarray,
        accel_x: np.ndarray,
        accel_y: np.ndarray,
        accel_z: np.ndarray,
        speeds_kmh: Optional[np.ndarray] = None,
        latitudes: Optional[np.ndarray] = None,
        longitudes: Optional[np.ndarray] = None,
        run_id: Optional[str] = None,
    ) -> PreprocessedData:
        """Compute vertical displacement from raw sensor arrays (for heavy_compute integration)."""
        rid = run_id or ""
        logger.info("Computing vertical displacement from raw sensor data", run_id=rid)
        
        # Create DataFrame from arrays
        df = pd.DataFrame({
            'time_ms': timestamps,
            'accel_x': accel_x,
            'accel_y': accel_y,
            'accel_z': accel_z,
            'gps_speed': speeds_kmh if speeds_kmh is not None else np.full_like(timestamps, 30.0),
            'lat': latitudes if latitudes is not None else np.full_like(timestamps, 0.0),
            'long': longitudes if longitudes is not None else np.full_like(timestamps, 0.0),
        })
        
        # Add dummy gravity columns for reorientation (if not available)
        if 'gravi_x' not in df.columns:
            # Assume phone is roughly level, gravity mostly in Z
            df['gravi_x'] = 0.0
            df['gravi_y'] = 0.0
            df['gravi_z'] = 9.81
        
        df['time_ms'] = timestamps * 1000  # convert to ms
        
        # Run preprocessing pipeline
        df = PreprocessingPipeline.run_pipeline(df)
        
        # Extract results
        vertical_disp = df['vertical_displacement'].to_numpy() if 'vertical_displacement' in df.columns else np.zeros(len(df))
        
        # Compute distances from speeds
        speeds_ms = df['gps_speed'].to_numpy() / 3.6
        dt = np.gradient(df['time_ms'].to_numpy() / 1000.0)
        dt = np.clip(dt, 0.001, 1.0)
        distances = np.cumsum(speeds_ms * dt)
        
        return PreprocessedData(
            timestamps=np.arange(len(df)) / 100.0,  # approximate 100Hz resampled
            accel_x=df['accel_x'].to_numpy(),
            accel_y=df['accel_y'].to_numpy(),
            accel_z=df['accel_z'].to_numpy(),
            speeds_kmh=df['gps_speed'].to_numpy(),
            latitudes=df['lat'].to_numpy(),
            longitudes=df['long'].to_numpy(),
            timestamps_ms=df['time_ms'].to_numpy(),
            vertical_displacement=df['vertical_displacement'].to_numpy(),
            distances_m=distances,
            total_distance_m=float(distances[-1]) if len(distances) > 0 else 0.0,
            sample_rate_hz=100.0,
        )


# Convenience function for backward compatibility
def compute_vertical_displacement(
    acceleration: np.ndarray,
    time_ms: np.ndarray,
    max_disp: float = 5.0,
) -> np.ndarray:
    """Compute vertical displacement from acceleration (standalone function).
    
    Reference implementation from ST5_Use_Vert_Displacement.py.
    Uses per-sample (non-accumulating) displacement calculation.
    """
    n = len(acceleration)
    if n < 3:
        return np.zeros(n)
    
    # Remove quadratic trend
    try:
        coeffs = np.polyfit(np.arange(n), acceleration, 2)
        trend = np.polyval(coeffs, np.arange(n))
        detrended = acceleration - trend
    except Exception:
        detrended = acceleration - np.mean(acceleration)
    
    dt = np.diff(time_ms / 1000.0)
    dt = np.insert(dt, 0, dt[0] if len(dt) > 0 else 0.01)
    dt = np.clip(dt, 0.001, 1.0)
    
    # Per-sample displacement (matching reference ST5_Use_Vert_Displacement.py)
    vertical_displacement = np.zeros(n)
    for i in range(1, n):
        abs_accel = np.abs(detrended[i]) / 10.0
        velocity = 0.5 * (abs_accel + np.abs(detrended[i - 1]) / 10.0) * dt[i]
        vertical_displacement[i] = 0.5 * (velocity + 0) * dt[i]
    
    return np.clip(vertical_displacement, -max_disp, max_disp)


if __name__ == "__main__":
    # Test with sample data
    import pandas as pd
    import numpy as np
    
    # Create sample vibration data
    n = 1000
    t = np.arange(n) * 0.01  # 100 Hz
    accel_z = np.random.normal(0, 0.5, n)
    time_ms = (t * 1000).astype(int)
    
    df = pd.DataFrame({
        'time_ms': time_ms,
        'accel_x': np.random.normal(0, 0.1, n),
        'accel_y': np.random.normal(0, 0.1, n),
        'accel_z': accel_z,
        'gravi_x': np.zeros(n),
        'gravi_y': np.zeros(n),
        'gravi_z': 9.81 * np.ones(n),
        'lat': np.full(n, 6.0),
        'long': np.full(n, -1.0),
        'gps_speed': np.full(n, 30.0),
        'time_ms': np.arange(n) * 10,
    })
    
    result = PreprocessingPipeline.run_pipeline(df)
    print(f"Processed {len(result)} rows")
    print(f"Vertical displacement range: {result['vertical_displacement'].min():.3f} to {result['vertical_displacement'].max():.3f} m")