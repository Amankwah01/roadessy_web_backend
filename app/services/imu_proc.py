"""
IMUProcessor - Process IMU/GPS data for pitch, roll, height per frame
Based on pothole-depth-fusion pipeline/imu_proc.py
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d


@dataclass
class IMUState:
    """IMU state for a single timestamp."""
    timestamp: float
    pitch_deg: float
    roll_deg: float
    yaw_deg: float
    height_m: float
    latitude: float
    longitude: float
    speed_kmh: float
    acceleration: np.ndarray  # [ax, ay, az] in m/s²


class IMUProcessor:
    """
    Process IMU/GPS data for pitch, roll, height per frame.
    Based on pothole-depth-fusion pipeline/imu_proc.py
    """
    
    def __init__(self, 
                 gravity: float = 9.81,
                 butterworth_order: int = 4,
                 cutoff_freq: float = 5.0):  # Hz
        self.gravity = gravity
        self.butterworth_order = butterworth_order
        self.cutoff_freq = cutoff_freq
        self._b_hp = None
        self._a_hp = None
        self._b_lp = None
        self._a_lp = None
    
    def process_imu_csv(self, 
                        csv_path: str,
                        target_fps: float = 30.0,
                        v_ref_kmh: float = 80.0) -> Dict[int, IMUState]:
        """
        Process IMU/GPS CSV file and return interpolated states per frame.
        
        Args:
            csv_path: Path to CSV with IMU/GPS data
            target_fps: Target frame rate for output
            v_ref_kmh: Reference speed for QCM (km/h)
            
        Returns:
            Dictionary mapping frame index to IMUState
        """
        # Load and preprocess CSV
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        
        # Parse timestamps
        time_col = self._find_time_column(df)
        df['time_s'] = self._parse_time_column(df[time_col])
        
        # Sort by time
        df = df.sort_values('time_s').reset_index(drop=True)
        
        # Parse accelerometer data
        accel_cols = self._find_accel_columns(df)
        accel_data = df[accel_cols].values.astype(np.float64)
        
        # Parse GPS data
        lat_col, lon_col = self._find_gps_columns(df)
        lat_data = df[lat_col].values if lat_col else np.zeros(len(df))
        lon_data = df[lon_col].values if lon_col else np.zeros(len(df))
        
        # Parse speed
        speed_col = self._find_speed_column(df)
        speed_data = df[speed_col].values if speed_col else np.full(len(df), 80.0/3.6)
        
        # Parse IMU orientation if available
        pitch_data, roll_data, yaw_data = self._parse_orientation(df)
        
        # Parse altitude/height
        height_data = self._parse_height(df)
        
        # Apply filters
        accel_filtered = self._filter_acceleration(accel_data, np.asarray(df['time_s'].values))
        orientation_filtered = self._filter_orientation(pitch_data, roll_data, yaw_data)
        height_filtered = self._filter_height(height_data)
        
        # Interpolate to target frame rate
        frame_times = np.arange(0, df['time_s'].iloc[-1], 1.0 / target_fps)
        imu_states = self._interpolate_to_frames(
            frame_times, df['time_s'].values,
            pitch_data, roll_data, yaw_data,
            height_filtered, lat_data, lon_data, speed_data
        )
        
        return imu_states
    
    def _find_time_column(self, df: pd.DataFrame) -> str:
        """Find timestamp column."""
        for col in ['Timestamp', 'timestamp', 'GPSTime', 'time', 'Time']:
            if col in df.columns:
                return col
        return df.columns[0]
    
    def _parse_time_column(self, series) -> np.ndarray:
        """Parse timestamp column to seconds."""
        try:
            # Try numeric first
            return pd.to_numeric(series, errors='raise').values
        except:
            # Try datetime
            parsed = pd.to_datetime(series, errors='coerce')
            if parsed.notna().any():
                return (parsed - parsed.iloc[0]).dt.total_seconds().values
            # Try numeric with errors='coerce'
            return pd.to_numeric(series, errors='coerce').fillna(0).values
    
    def _find_accel_columns(self, df: pd.DataFrame) -> List[str]:
        """Find accelerometer columns."""
        candidates = [
            ['LinAccelX', 'LinAccelY', 'LinAccelZ'],
            ['AccelX', 'AccelY', 'AccelZ'],
            ['accel_x', 'accel_y', 'accel_z'],
            ['ax', 'ay', 'az']
        ]
        for cols in candidates:
            if all(c in df.columns for c in cols):
                return list(cols)
        # Fallback: first 3 numeric columns after time
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return numeric_cols[:3]
    
    def _find_gps_columns(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """Find latitude and longitude columns."""
        lat_candidates = ['Latitude', 'latitude', 'lat', 'Lat', 'Lat', 'LAT']
        lon_candidates = ['Longitude', 'longitude', 'lon', 'Lon', 'LON']
        
        lat_col = next((c for c in lat_candidates if c in df.columns), None)
        lon_col = next((c for c in lon_candidates if c in df.columns), None)
        return lat_col, lon_col
    
    def _find_speed_column(self, df: pd.DataFrame) -> Optional[str]:
        for col in ['Speed', 'speed', 'SpeedKmh', 'SpeedKmh', 'GPS_Speed', 'GpsSpeed']:
            if col in df.columns:
                return col
        return None
    
    def _parse_orientation(self, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Parse pitch, roll, yaw from dataframe."""
        pitch_cols = ['Pitch', 'pitch', 'PitchDeg', 'pitch_deg']
        roll_cols = ['Roll', 'roll', 'RollDeg', 'roll_deg']
        yaw_cols = ['Yaw', 'yaw', 'YawDeg', 'yaw_deg', 'Heading', 'heading']
        
        pitch = self._get_column_data(df, pitch_cols)
        roll = self._get_column_data(df, roll_cols)
        yaw = self._get_column_data(df, yaw_cols)
        return pitch, roll, yaw
    
    def _get_column_data(self, df: pd.DataFrame, candidates: List[str]) -> Optional[np.ndarray]:
        for col in candidates:
            if col in df.columns:
                return np.asarray(pd.to_numeric(df[col], errors='coerce').fillna(0).values)
        return None
    
    def _parse_height(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Parse height/altitude from dataframe."""
        for col in ['Altitude', 'altitude', 'Height', 'height', 'GPSAltitude', 'GPSAlt']:
            if col in df.columns:
                return np.asarray(pd.to_numeric(df[col], errors='coerce').fillna(1.5).values)
        return None
    
    def _filter_acceleration(self, accel: np.ndarray, time_s: np.ndarray) -> np.ndarray:
        """Apply high-pass and low-pass filters to acceleration."""
        dt = np.median(np.diff(time_s))
        fs = 1.0 / dt if dt > 0 else 100.0
        
        # High-pass filter to remove gravity
        if self._b_hp is None:
            self._design_filters(fs)
        
        # If filters couldn't be designed (e.g., hp_cutoff >= 1.0), return original
        if self._b_hp is None or self._a_hp is None:
            return accel
        
        accel_hp = np.zeros_like(accel)
        for i in range(3):
            accel_hp[:, i] = filtfilt(self._b_hp, self._a_hp, accel[:, i])
        
        return accel_hp
    
    def _design_filters(self, fs: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        nyq = 0.5 * fs
        hp_cutoff = 0.1 / nyq
        if hp_cutoff < 1.0:
            self._b_hp, self._a_hp = butter(self.butterworth_order, hp_cutoff, btype='highpass')  # type: ignore[return-value, assignment]
        else:
            self._b_hp, self._a_hp = None, None
        return self._b_hp, self._a_hp  # type: ignore[return-value]
    
    def _filter_orientation(self, pitch, roll, yaw) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Low-pass filter orientation data."""
        # Simple moving average as fallback
        window = 5
        def smooth(x):
            if x is None:
                return None
            return np.convolve(x, np.ones(window)/window, mode='same')
        return smooth(pitch), smooth(roll), smooth(yaw)  # type: ignore[return-value]
    
    def _filter_height(self, height):
        """Smooth height data."""
        if height is None:
            return np.full(100, 1.5)
        return np.convolve(height, np.ones(5)/5, mode='same')
    
    def _interpolate_to_frames(self, frame_times, imu_times,
                               pitch, roll, yaw, height, lat, lon, speed) -> Dict[int, IMUState]:
        """Interpolate IMU data to frame timestamps."""
        imu_states = {}
        
        # Create interpolators
        pitch_interp = interp1d(imu_times, pitch, kind='linear', bounds_error=False, fill_value=(pitch[0], pitch[-1])) if pitch is not None else None  # type: ignore[arg-type]
        roll_interp = interp1d(imu_times, roll, kind='linear', bounds_error=False, fill_value=(roll[0], roll[-1])) if roll is not None else None  # type: ignore[arg-type]
        yaw_interp = interp1d(imu_times, yaw, kind='linear', bounds_error=False, fill_value=(yaw[0], yaw[-1])) if yaw is not None else None  # type: ignore[arg-type]
        height_interp = interp1d(imu_times, height, kind='linear', bounds_error=False, fill_value=(height[0], height[-1])) if height is not None else None  # type: ignore[arg-type]
        lat_interp = interp1d(imu_times, lat, kind='linear', bounds_error=False, fill_value=(lat[0], lat[-1])) if lat is not None else None  # type: ignore[arg-type]
        lon_interp = interp1d(imu_times, lon, kind='linear', bounds_error=False, fill_value=(lon[0], lon[-1])) if lon is not None else None  # type: ignore[arg-type]
        speed_interp = interp1d(imu_times, speed, kind='linear', bounds_error=False, fill_value=(speed[0], speed[-1])) if speed is not None else None  # type: ignore[arg-type]
        
        for i, t in enumerate(frame_times):
            pitch_val = float(pitch_interp(t)) if pitch_interp else 0.0
            roll_val = float(roll_interp(t)) if roll_interp else 0.0
            yaw_val = float(yaw_interp(t)) if yaw_interp else 0.0
            height_val = float(height_interp(t)) if height_interp else 1.5
            lat_val = float(lat_interp(t)) if lat_interp else 0.0
            lon_val = float(lon_interp(t)) if lon_interp else 0.0
            speed_val = float(speed_interp(t)) if speed_interp else 80.0/3.6
            
            imu_states[i] = IMUState(
                timestamp=t,
                pitch_deg=pitch_val,
                roll_deg=roll_val,
                yaw_deg=yaw_val,
                height_m=height_val,
                latitude=lat_val,
                longitude=lon_val,
                speed_kmh=speed_val * 3.6,
                acceleration=np.array([0.0, 0.0, 9.81])  # Placeholder
            )
        
        return imu_states
    
    def _ema(self, prev: float, new: float) -> float:
        """Exponential moving average."""
        return prev * (1 - 0.35) + new * 0.35


def process_imu_csv(csv_path: str, target_fps: float = 30.0, v_ref_kmh: float = 80.0) -> Dict[int, IMUState]:
    """Convenience function to process IMU CSV and return frame-indexed states."""
    processor = IMUProcessor()
    return processor.process_imu_csv(csv_path, target_fps, v_ref_kmh)