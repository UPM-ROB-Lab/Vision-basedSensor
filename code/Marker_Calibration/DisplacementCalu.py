#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import chardet
import traceback
import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
from pathlib import Path

# ------------------------------
# Configuration using dataclass for type safety
# ------------------------------
@dataclass
class Config:
    """Configuration parameters for marker tracking and analysis"""
    marker_diameter_mm: float = 2.0  # Physical diameter of markers
    warmup_frames: int = 100         # Camera warmup frames to skip
    min_marker_size_px: float = 5.0  # Minimum detectable marker size
    max_displacement_px: float = 50.0 # Maximum allowed frame-to-frame movement
    data_dir: Path = Path("Results/data")
    output_dir: Path = Path("Results/data/results")
    plots_dir: Path = Path("Results/data/results/Displacement_Analysis_Plots")
    column_mapping: Dict[str, str] = {
        'Cx': 'u',                   # X-coordinate column
        'Cy': 'v',                   # Y-coordinate column 
        'major_axis': 'major_axis'   # Marker major axis column
    }

# Initialize configuration
CONFIG = Config()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG.output_dir / 'analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CameraParameters:
    """Container for camera intrinsic and extrinsic parameters"""
    def __init__(self):
        self.matrix: np.ndarray = None      # Camera matrix (3x3)
        self.dist_coeffs: np.ndarray = None # Distortion coefficients
        self.R_world_to_cam: np.ndarray = None # Rotation matrix (world to cam)
        self.T_world_to_cam: np.ndarray = None # Translation vector (world to cam)
        self.resolution: Tuple[int, int] = None # Camera resolution

class MarkerAnalysis:
    """Main class for marker tracking and 3D displacement analysis"""
    
    def __init__(self, config: Config):
        self.config = config
        self.camera = CameraParameters()
        self._validate_paths()
        
    def _validate_paths(self) -> None:
        """Ensure required directories exist"""
        for path in [self.config.data_dir, self.config.output_dir, self.config.plots_dir]:
            path.mkdir(parents=True, exist_ok=True)
            
    def load_parameters(self, intrinsic_path: Path, extrinsic_path: Path) -> None:
        """
        Load camera calibration parameters with comprehensive validation
        
        Args:
            intrinsic_path: Path to intrinsic parameters file
            extrinsic_path: Path to extrinsic parameters file
            
        Raises:
            ValueError: If parameters are invalid or files can't be read
        """
        try:
            # Load intrinsic parameters with validation
            intrinsic_df = pd.read_excel(intrinsic_path)
            params = intrinsic_df.set_index('Parameter')['Value']
            
            # Build camera matrix with validation
            self.camera.matrix = np.array([
                [params['fx'], params.get('skew', 0), params['cx']],
                [0, params['fy'], params['cy']],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Validate positive focal lengths
            if self.camera.matrix[0,0] <= 0 or self.camera.matrix[1,1] <= 0:
                raise ValueError("Focal lengths must be positive")
                
            # Load distortion coefficients
            dist_keys = ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6']
            self.camera.dist_coeffs = np.array(
                [params.get(k, 0) for k in dist_keys[:5]],  # OpenCV uses 5 by default
                dtype=np.float32
            )
            
            # Load extrinsic parameters with validation
            extrinsic_df = pd.read_excel(extrinsic_path)
            ext_params = extrinsic_df.set_index('Parameter')['Value']
            
            # Rotation matrix validation
            self.camera.R_world_to_cam = np.array([
                [ext_params[f'R_wc_{i}{j}'] for j in range(1,4)] 
                for i in range(1,4)
            ], dtype=np.float32)
            
            # Verify rotation matrix is orthogonal
            if not np.allclose(self.camera.R_world_to_cam @ self.camera.R_world_to_cam.T, 
                              np.eye(3), atol=1e-6):
                raise ValueError("Rotation matrix is not orthogonal")
                
            # Translation vector
            self.camera.T_world_to_cam = np.array([
                ext_params['Tx_wc'], 
                ext_params['Ty_wc'], 
                ext_params['Tz_wc']
            ], dtype=np.float32).reshape(3, 1)
            
            logger.info("Camera parameters loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load camera parameters: {str(e)}")
            raise
            
    def load_marker_data(self, filepath: Path) -> pd.DataFrame:
        """
        Load marker tracking data with robust parsing and validation
        
        Args:
            filepath: Path to marker data CSV file
            
        Returns:
            Cleaned and validated DataFrame of marker positions
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If data is invalid or missing required columns
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Marker data file not found: {filepath}")
            
        try:
            # Detect encoding and handle various delimiters
            with open(filepath, 'rb') as f:
                encoding = chardet.detect(f.read(30000))['encoding'] or 'utf-8'
                
            df = pd.read_csv(
                filepath, 
                sep=r'\s+|,|\t', 
                encoding=encoding, 
                engine='python', 
                skipinitialspace=True
            )
            
            # Clean column names and validate
            df.columns = [col.strip() for col in df.columns]
            required_cols = set(self.config.column_mapping.keys()) | {'frameno', 'row', 'col'}
            missing = required_cols - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
                
            # Apply column mapping and basic validation
            df = df.rename(columns=self.config.column_mapping)
            
            # Filter out invalid markers (too small)
            valid = df['major_axis'] >= self.config.min_marker_size_px
            if not valid.all():
                logger.warning(f"Filtered {len(df) - valid.sum()} markers for being too small")
                df = df[valid].copy()
                
            # Sort by frame number for sequential processing
            return df.sort_values('frameno').reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Failed to load marker data: {str(e)}")
            raise
            
    def _undistort_points(self, points: np.ndarray) -> np.ndarray:
        """Apply camera distortion correction to 2D points"""
        return cv2.undistortPoints(
            points.reshape(-1, 1, 2),
            self.camera.matrix,
            self.camera.dist_coeffs,
            None,
            self.camera.matrix
        ).reshape(-1, 2)
        
    def _calculate_3d_position(self, u: float, v: float, diameter_px: float) -> np.ndarray:
        """
        Calculate 3D world position of a marker from image coordinates
        
        Args:
            u, v: Undistorted image coordinates
            diameter_px: Observed marker diameter in pixels
            
        Returns:
            3D position in world coordinates as numpy array
            
        Raises:
            ValueError: If calculation produces invalid results
        """
        fx, fy = self.camera.matrix[0, 0], self.camera.matrix[1, 1]
        cx, cy = self.camera.matrix[0, 2], self.camera.matrix[1, 2]
        f_avg = (fx + fy) / 2
        
        try:
            # Calculate depth using perspective projection
            R = np.sqrt((u - cx)**2 + (v - cy)**2)
            if R < 1e-6:
                raise ValueError("Marker too close to principal point")
                
            d_effective = (self.config.marker_diameter_mm / f_avg) * np.sqrt(R**2 + f_avg**2)
            h = f_avg * (d_effective / diameter_px)
            
            # Convert to 3D camera coordinates
            Xc = h * (u - cx) / fx
            Yc = h * (v - cy) / fy
            P_cam = np.array([Xc, Yc, h]).reshape(3, 1)
            
            # Transform to world coordinates
            P_world = (self.camera.R_world_to_cam.T @ (P_cam - self.camera.T_world_to_cam)).flatten()
            
            # Sanity check results
            if not np.all(np.isfinite(P_world)):
                raise ValueError("Non-finite coordinates calculated")
                
            return P_world
            
        except Exception as e:
            logger.warning(f"3D calculation failed: {str(e)}")
            raise
            
    def _track_markers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform marker tracking and 3D displacement calculation
        
        Args:
            df: DataFrame of marker positions
            
        Returns:
            DataFrame with calculated 3D positions and displacements
        """
        # Initialize results storage
        results = []
        marker_dict = {}
        
        # Skip warmup frames if specified
        if self.config.warmup_frames > 0:
            df = df[df['frameno'] >= df['frameno'].min() + self.config.warmup_frames].copy()
            
        # Undistort all points at once for efficiency
        points = df[['u', 'v']].values
        df[['u', 'v']] = self._undistort_points(points)
        
        # Process each frame in sequence
        for frame_num, frame_data in df.groupby('frameno'):
            current_markers = {}
            
            for _, row in frame_data.iterrows():
                marker_key = (row['row'], row['col'])  # Unique identifier
                
                # Store current marker info
                current_markers[marker_key] = {
                    'u': row['u'],
                    'v': row['v'],
                    'diameter_px': row['major_axis']
                }
                
                # Calculate 3D position if we have previous data
                if marker_key in marker_dict:
                    try:
                        prev_pos = self._calculate_3d_position(
                            marker_dict[marker_key]['u'],
                            marker_dict[marker_key]['v'],
                            marker_dict[marker_key]['diameter_px']
                        )
                        
                        curr_pos = self._calculate_3d_position(
                            row['u'], row['v'], row['major_axis']
                        )
                        
                        displacement = curr_pos - prev_pos
                        displacement_mm = np.linalg.norm(displacement)
                        
                        # Validate displacement isn't too large
                        if displacement_mm > self.config.max_displacement_px:
                            raise ValueError(f"Displacement too large: {displacement_mm:.2f}mm")
                            
                        results.append({
                            'frameno': frame_num,
                            'row': row['row'],
                            'col': row['col'],
                            'X': curr_pos[0],
                            'Y': curr_pos[1],
                            'Z': curr_pos[2],
                            'dX': displacement[0],
                            'dY': displacement[1],
                            'dZ': displacement[2],
                            'displacement': displacement_mm
                        })
                        
                    except Exception as e:
                        logger.debug(f"Frame {frame_num} marker {marker_key}: {str(e)}")
                        continue
                        
            # Update marker dictionary with current frame
            marker_dict.update(current_markers)
            
        return pd.DataFrame(results)
        
    def analyze_displacement(self, results_df: pd.DataFrame) -> None:
        """
        Generate comprehensive displacement analysis and visualizations
        
        Args:
            results_df: DataFrame containing 3D marker positions and displacements
        """
        if results_df.empty:
            logger.warning("No valid displacement data to analyze")
            return
            
        # Ensure output directory exists
        self.config.plots_dir.mkdir(exist_ok=True)
        
        # Calculate cumulative displacements
        results_df = results_df.sort_values(['row', 'col', 'frameno'])
        results_df['cumulative_displacement'] = results_df.groupby(['row', 'col'])['displacement'].cumsum()
        
        # Generate plots for each marker
        for (row, col), marker_data in results_df.groupby(['row', 'col']):
            fig, axes = plt.subplots(3, 1, figsize=(12, 12))
            
            # Plot 3D trajectory
            ax = axes[0]
            ax = fig.add_subplot(2, 2, (1, 3), projection='3d')
            ax.plot(
                marker_data['X'], 
                marker_data['Y'], 
                marker_data['Z'], 
                'b.-',
                linewidth=0.5,
                markersize=3
            )
            ax.set(
                title=f'3D Trajectory - Marker ({row}, {col})',
                xlabel='X (mm)',
                ylabel='Y (mm)',
                zlabel='Z (mm)'
            )
            
            # Plot frame-to-frame displacement
            ax2 = axes[1]
            ax2.plot(
                marker_data['frameno'],
                marker_data['displacement'],
                'r.-',
                markersize=3
            )
            ax2.set(
                title='Frame-to-Frame Displacement',
                xlabel='Frame Number',
                ylabel='Displacement (mm)',
                ylim=(0, None)
            )
            ax2.grid(True)
            
            # Plot cumulative displacement
            ax3 = axes[2]
            ax3.plot(
                marker_data['frameno'],
                marker_data['cumulative_displacement'],
                'g.-',
                markersize=3
            )
            ax3.set(
                title='Cumulative Displacement',
                xlabel='Frame Number',
                ylabel='Total Displacement (mm)',
                ylim=(0, None)
            )
            ax3.grid(True)
            
            plt.tight_layout()
            plot_path = self.config.plots_dir / f'marker_{row}_{col}_analysis.png'
            plt.savefig(plot_path, dpi=150)
            plt.close()
            logger.info(f"Saved analysis plot: {plot_path}")
            
        # Save summary statistics
        stats = results_df.groupby(['row', 'col']).agg({
            'displacement': ['mean', 'std', 'max'],
            'cumulative_displacement': 'last'
        })
        stats_path = self.config.plots_dir / 'displacement_statistics.csv'
        stats.to_csv(stats_path)
        logger.info(f"Saved displacement statistics: {stats_path}")
        
    def run_analysis(self, input_csv: Path) -> None:
        """
        Complete analysis pipeline from marker data to 3D results
        
        Args:
            input_csv: Path to input marker data CSV file
        """
        try:
            logger.info("Starting marker analysis pipeline")
            
            # Load data
            intrinsic_path = self.config.data_dir / 'PreprocessPara' / 'IntrinsicParameters.xlsx'
            extrinsic_path = self.config.data_dir / 'PreprocessPara' / 'ExtrinsicParameters.xlsx'
            self.load_parameters(intrinsic_path, extrinsic_path)
            
            marker_df = self.load_marker_data(input_csv)
            logger.info(f"Loaded {len(marker_df)} marker detections")
            
            # Perform tracking and 3D calculation
            results_df = self._track_markers(marker_df)
            if results_df.empty:
                raise ValueError("No valid 3D positions calculated")
                
            logger.info(f"Calculated 3D positions for {len(results_df)} marker observations")
            
            # Save results
            output_path = self.config.output_dir / 'marker_3d_coordinates.xlsx'
            results_df.to_excel(output_path, index=False)
            logger.info(f"Saved 3D coordinates to {output_path}")
            
            # Generate analysis plots
            self.analyze_displacement(results_df)
            logger.info("Analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
if __name__ == "__main__":
    try:
        # Initialize and run analysis
        analyzer = MarkerAnalysis(CONFIG)
        input_csv = CONFIG.data_dir / 'marker_locations_0.csv'
        analyzer.run_analysis(input_csv)
        
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        exit(1)