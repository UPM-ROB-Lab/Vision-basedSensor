#!/usr/bin/env python3
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Dict, Tuple, Optional, List

# =============================================================================
# SCI-STYLE PLOT CONFIGURATION
# =============================================================================
plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'font.size': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'savefig.dpi': 300,
    'figure.autolayout': True  # Auto tight_layout
})

# =============================================================================
# DATA STRUCTURES AND CONSTANTS
# =============================================================================
class CameraParameters:
    """Container for camera intrinsic and extrinsic parameters"""
    def __init__(self):
        self.matrix = None
        self.distortion = None
        self.R_world_to_cam = None
        self.T_world_to_cam = None
        self.reprojection_error = None

MARKER_DIAMETER_MM = 2.0  # Circle marker diameter in mm

# =============================================================================
# CORE FUNCTIONS
# =============================================================================
def load_intrinsics_from_excel(filepath: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load camera intrinsic parameters from Excel file with validation."""
    try:
        df = pd.read_excel(filepath)
        params = df.set_index('Parameter')['Value']
        
        # Build camera matrix
        camera_matrix = np.array([
            [params['fx'], params.get('skew', 0), params['cx']],
            [0, params['fy'], params['cy']],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Build distortion coefficients
        dist_order = 5  # Maximum distortion order to check
        dist_coeffs = np.array([
            params.get(f'k{i+1}', 0) for i in range(dist_order)
        ] + [
            params.get(f'p{i+1}', 0) for i in range(2)
        ], dtype=np.float32)
        
        print("Successfully loaded intrinsic parameters")
        return camera_matrix, dist_coeffs
        
    except FileNotFoundError:
        print(f"Error: Intrinsic parameters file not found at '{filepath}'")
        return None, None
    except KeyError as e:
        print(f"Error: Missing required parameter in Excel file: {e}")
        return None, None
    except Exception as e:
        print(f"Unexpected error loading intrinsics: {str(e)}")
        return None, None

def calibrate_camera_extrinsics(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """Calculate camera extrinsic parameters using PnP algorithm."""
    # Input validation and reshaping
    if object_points.shape[0] < 4:
        print("Need at least 4 points for PnP")
        return None, None, None
    
    object_points = np.ascontiguousarray(object_points.reshape(-1, 1, 3), dtype=np.float32)
    image_points = np.ascontiguousarray(image_points.reshape(-1, 1, 2), dtype=np.float32)
    
    # Run PnP RANSAC
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
        confidence=0.99,
        reprojectionError=8.0,
        iterationsCount=1000
    )
    
    if not success:
        print("PnP failed to find solution")
        return None, None, None
    
    # Convert rotation vector to matrix
    R, _ = cv2.Rodrigues(rvec)
    T = tvec.reshape(3, 1)
    
    # Calculate reprojection error
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    error = np.mean(np.linalg.norm(projected - image_points, axis=2))
    
    print(f"PnP solved with {len(inliers)} inliers")
    print(f"Mean reprojection error: {error:.3f} pixels")
    
    return R, T, error

def save_extrinsics_to_excel(
    R: np.ndarray,
    T: np.ndarray,
    error: float,
    filepath: str,
    description: str = ""
) -> bool:
    """Save extrinsic parameters to Excel file with comprehensive metadata."""
    try:
        # Prepare data
        data = [
            ["--- Camera Extrinsic Parameters ---", "", ""],
            ["Calibration Date", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), ""],
            ["Reprojection Error (px)", error, ""],
            ["", "", ""],
            ["--- World to Camera Transformation ---", "", ""]
        ]
        
        # Add rotation matrix elements
        for i in range(3):
            for j in range(3):
                data.append([f"R_wc_{i+1}{j+1}", R[i, j], f"Rotation matrix element ({i+1},{j+1})"])
        
        # Add translation vector
        T_flat = T.flatten()
        for i, axis in enumerate(['X', 'Y', 'Z']):
            data.append([f"T_wc_{axis}", T_flat[i], f"Translation in {axis}-axis (mm)"])
        
        # Save to DataFrame and Excel
        df = pd.DataFrame(data, columns=["Parameter", "Value", "Description"])
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_excel(filepath, index=False)
        print(f"Extrinsic parameters saved to {filepath}")
        return True
    except Exception as e:
        print(f"Failed to save extrinsics: {str(e)}")
        return False

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def plot_3d_calibration_result(
    world_points: np.ndarray,
    R_wc: np.ndarray,
    T_wc: np.ndarray,
    title: str = "Extrinsic Calibration Result"
) -> None:
    """Create 3D visualization of calibration results with camera frustum."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot world points
    ax.scatter(
        world_points[:, 0], world_points[:, 1], world_points[:, 2],
        c='steelblue', marker='o', s=40, alpha=0.8,
        label='Control Points'
    )
    
    # Calculate camera position in world coordinates
    R_cw = R_wc.T
    cam_pos = -R_cw @ T_wc
    
    # Create camera frustum
    scale = np.ptp(world_points) * 0.2  # Dynamic scaling
    frustum = np.array([
        [0, 0, 0],
        [-1, -1, 2], [1, -1, 2],
        [1, 1, 2], [-1, 1, 2]
    ]) * scale
    
    # Transform frustum to world coordinates
    frustum_world = (R_cw @ frustum.T + cam_pos).T
    
    # Create frustum faces
    faces = [
        [frustum_world[0], frustum_world[1], frustum_world[2]],
        [frustum_world[0], frustum_world[2], frustum_world[3]],
        [frustum_world[0], frustum_world[3], frustum_world[4]],
        [frustum_world[0], frustum_world[4], frustum_world[1]],
        frustum_world[1:]
    ]
    
    # Plot frustum
    ax.add_collection3d(Poly3DCollection(
        faces,
        facecolors='crimson',
        edgecolors='darkred',
        alpha=0.25,
        linewidths=1
    ))
    
    # Plot camera position
    ax.scatter(
        *cam_pos.flatten(),
        c='red', marker='s', s=100,
        label='Camera Position'
    )
    
    # Plot world origin
    ax.scatter(
        0, 0, 0,
        c='black', marker='x', s=100,
        label='World Origin'
    )
    
    # Axis labels and title
    ax.set_xlabel('X (mm)', fontweight='bold', labelpad=10)
    ax.set_ylabel('Y (mm)', fontweight='bold', labelpad=10)
    ax.set_zlabel('Z (mm)', fontweight='bold', labelpad=10)
    ax.set_title(title, fontsize=14, pad=20)
    
    # Equal aspect ratio
    set_3d_axes_equal(ax)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

def set_3d_axes_equal(ax):
    """Set 3D axes to equal scale."""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    center = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([center[0] - radius, center[0] + radius])
    ax.set_ylim3d([center[1] - radius, center[1] + radius])
    ax.set_zlim3d([center[2] - radius, center[2] + radius])

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    # Configuration
    DATA_DIR = os.path.join("Results", "data", "PreprocessPara")
    PATHS = {
        'intrinsic': os.path.join(DATA_DIR, "IntrinsicParameters.xlsx"),
        'extrinsic': os.path.join(DATA_DIR, "ExtrinsicParameters.xlsx"),
        'world_points': os.path.join(DATA_DIR, "world_marker_CMM.csv"),
        'image_points': os.path.join(DATA_DIR, "pixel_marker.csv")
    }
    
    # Load camera intrinsics
    K, dist = load_intrinsics_from_excel(PATHS['intrinsic'])
    if K is None:
        return
    
    # Load correspondences
    try:
        df_world = pd.read_csv(PATHS['world_points'])
        df_image = pd.read_csv(PATHS['image_points'])
        
        # Merge on marker_id
        df_merged = pd.merge(
            df_world, df_image,
            on='marker_id',
            suffixes=('_world', '_image')
        )
        
        # Prepare 3D-2D correspondences
        object_points = df_merged[['Xw', 'Yw', 'Zw']].values.astype(np.float32)
        image_points = df_merged[['u', 'v']].values.astype(np.float32)
        
    except Exception as e:
        print(f"Error loading correspondences: {str(e)}")
        return
    
    # Run calibration
    R, T, error = calibrate_camera_extrinsics(object_points, image_points, K, dist)
    if R is None:
        return
    
    # Save results
    save_extrinsics_to_excel(R, T, error, PATHS['extrinsic'])
    
    # Visualization
    plot_3d_calibration_result(
        object_points,
        R, T,
        "Camera Extrinsic Calibration\n(World Coordinate Frame)"
    )

if __name__ == "__main__":
    main()