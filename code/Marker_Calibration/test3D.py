import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os
from matplotlib.ticker import FormatStrFormatter

# ------------------------------
# Configuration Parameters
# ------------------------------
REAL_MARKER_DIAMETER_MM = 2.0  # Diameter of circular markers
DPI = 400  # High resolution for publication quality
MARKER_SIZE = 80  # Base marker size
FONT_SIZE = 12  # Base font size

# =============================================================================
# Core Utility Functions
# =============================================================================
def load_intrinsics_from_excel(filepath):
    """Load camera intrinsic parameters from Excel file"""
    try:
        df = pd.read_excel(filepath)
        params = df.set_index('Parameter')['Value']
        camera_matrix = np.array([
            [params['fx'], params.get('skew', 0), params['cx']],
            [0,             params['fy'],          params['cy']],
            [0,             0,                     1]
        ], dtype=np.float32)
        
        dist_keys = ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6']
        dist_coeffs = np.array([params.get(key, 0) for key in dist_keys], dtype=np.float32)
        return camera_matrix, dist_coeffs
    except Exception as e:
        print(f"Error loading intrinsics: {str(e)}")
        return None, None

def load_extrinsics_from_excel(filepath):
    """Load camera extrinsic parameters from Excel file"""
    try:
        df = pd.read_excel(filepath)
        params = df.set_index('Parameter')['Value']
        R_world_to_cam = np.array([
            [params['R_wc_11'], params['R_wc_12'], params['R_wc_13']],
            [params['R_wc_21'], params['R_wc_22'], params['R_wc_23']],
            [params['R_wc_31'], params['R_wc_32'], params['R_wc_33']]
        ], dtype=np.float32)
        
        T_world_to_cam = np.array([
            params['Tx_wc'], params['Ty_wc'], params['Tz_wc']
        ], dtype=np.float32).reshape(3, 1)
        return R_world_to_cam, T_world_to_cam
    except Exception as e:
        print(f"Error loading extrinsics: {str(e)}")
        return None, None

def calculate_3d_point_twostep(pixel_coord, pixel_major_axis, camera_matrix, dist_coeffs):
    """Calculate 3D camera coordinates from 2D pixel information using two-step method"""
    if pixel_major_axis < 1e-6: 
        return None
        
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    f_avg = (fx + fy) / 2.0
    
    pixel_reshaped = np.array([[pixel_coord]], dtype=np.float32)
    undistorted_pixel = cv2.undistortPoints(pixel_reshaped, camera_matrix, dist_coeffs, 
                                          None, camera_matrix)[0,0]
    
    u_undist, v_undist = undistorted_pixel[0], undistorted_pixel[1]
    R_px = np.sqrt((u_undist - cx)**2 + (v_undist - cy)**2)
    
    # Effective diameter calculation
    d_effective = (REAL_MARKER_DIAMETER_MM / f_avg) * np.sqrt(R_px**2 + f_avg**2)
    
    # Depth calculation
    Zc = f_avg * (d_effective / pixel_major_axis)
    
    # 3D coordinates
    Xc = Zc * (u_undist - cx) / fx
    Yc = Zc * (v_undist - cy) / fy
    
    return np.array([Xc, Yc, Zc])

def create_3d_comparison_plot(calc_points, truth_points, errors, output_path):
    """Create publication-quality 3D comparison plot"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot elements with enhanced styling
    ax.scatter(
        calc_points[:, 0], calc_points[:, 1], calc_points[:, 2],
        c='#1f77b4', marker='o', s=MARKER_SIZE, 
        edgecolors='k', linewidth=0.5, alpha=0.7,
        label='Calculated Pose'
    )
    
    ax.scatter(
        truth_points[:, 0], truth_points[:, 1], truth_points[:, 2],
        c='#2ca02c', marker='^', s=MARKER_SIZE+20,
        alpha=0.8, label='Ground Truth'
    )
    
    # Plot error vectors
    for i in range(len(calc_points)):
        ax.plot(
            [calc_points[i,0], truth_points[i,0]], 
            [calc_points[i,1], truth_points[i,1]], 
            [calc_points[i,2], truth_points[i,2]], 
            color='#d62728', linestyle='-', linewidth=1.5, alpha=0.7
        )
        
        # Add marker ID labels
        ax.text(truth_points[i,0], truth_points[i,1], truth_points[i,2] + errors[i] * 0.1, 
               f"{i+1}", color='black', fontsize=FONT_SIZE-1, weight='bold')
    
    # Set labels and title
    ax.set_xlabel('X (mm)', fontsize=FONT_SIZE, labelpad=10)
    ax.set_ylabel('Y (mm)', fontsize=FONT_SIZE, labelpad=10)
    ax.set_zlabel('Z (mm)', fontsize=FONT_SIZE, labelpad=10)
    
    mean_error = np.mean(errors)
    title = f'3D Pose Comparison\nMean Error: {mean_error:.2f} mm'
    ax.set_title(title, fontsize=FONT_SIZE+2, pad=20)
    
    ax.legend(fontsize=FONT_SIZE-1, loc='upper right')
    ax.grid(False)
    
    # Set equal aspect ratio
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    return fig

def create_error_analysis_plot(marker_ids, errors, output_path):
    """Create publication-quality error analysis plot"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
    fig.suptitle('Per-Marker Error Analysis', fontsize=24, weight='bold')
    
    error_components = [
        (errors[:, 0], 'X-axis Error', '#d62728'),
        (errors[:, 1], 'Y-axis Error', '#2ca02c'), 
        (errors[:, 2], 'Z-axis Error', '#1f77b4'),
        (np.linalg.norm(errors, axis=1), 'Total Error', '#7f7f7f')
    ]
    
    for i, (err_data, title, color) in enumerate(error_components):
        ax = axs.flat[i]
        mean_err = np.mean(np.abs(err_data))
        
        ax.plot(marker_ids, np.abs(err_data), 'o-', 
               color=color, linewidth=2, markersize=8)
        ax.axhline(mean_err, color=color, linestyle=':', 
                  linewidth=1.5, label=f'Mean: {mean_err:.2f} mm')
        
        ax.set_title(title, fontsize=18, weight='bold')
        ax.set_ylabel('Error (mm)', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(False)
        ax.tick_params(labelsize=12)
        
        if i >= 2:
            ax.set_xlabel('Marker ID', fontsize=14)
            ax.set_xticks(marker_ids)
            ax.set_xticklabels(marker_ids, rotation=45, ha='right')
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    return fig

# =============================================================================
# Main Program
# =============================================================================
def main():
    # Configure paths
    base_dir = os.path.join("Results", "data", "PreprocessPara")
    raw_marker_txt = os.path.join(base_dir, 'MarkerCalibration.txt')
    intrinsic_file = os.path.join(base_dir, "IntrinsicParameters.xlsx")
    extrinsic_file = os.path.join(base_dir, "ExtrinsicParameters.xlsx")
    ground_truth_file = os.path.join(base_dir, "world_marker_CMM.csv")
    
    output_3d = os.path.join(base_dir, '3d_pose_comparison.png')
    output_error = os.path.join(base_dir, 'error_analysis.png')
    
    try:
        # 1. Load and process data
        print("Loading and processing data...")
        df = pd.read_csv(raw_marker_txt, header=None, skiprows=1, sep='\s+', 
                        usecols=[0, 1, 2, 5, 6, 7],
                        names=['frameno', 'row', 'col', 'u', 'v', 'major_axis'], 
                        engine='python')
        
        # Clean and process data
        for col in df.columns:
            if df[col].dtype == 'object': 
                df[col] = df[col].str.strip(',')
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        
        # Group by marker positions
        pixel_df = df.groupby(['row', 'col'])[['u', 'v', 'major_axis']].mean().reset_index()
        pixel_df['marker_id'] = range(1, len(pixel_df) + 1)
        
        # Load camera parameters
        camera_matrix, dist_coeffs = load_intrinsics_from_excel(intrinsic_file)
        R_wc, T_wc = load_extrinsics_from_excel(extrinsic_file)
        
        if None in [camera_matrix, R_wc]:
            raise ValueError("Failed to load camera parameters")
        
        # Load ground truth
        truth_df = pd.read_csv(ground_truth_file)
        truth_df = truth_df.drop_duplicates('marker_id')
        
        # 2. Calculate 3D coordinates
        print("Calculating 3D coordinates...")
        world_coords = []
        for _, row in pixel_df.iterrows():
            point_3d = calculate_3d_point_twostep(
                (row['u'], row['v']), 
                row['major_axis'], 
                camera_matrix, 
                dist_coeffs
            )
            if point_3d is not None:
                world_coords.append({
                    'marker_id': row['marker_id'],
                    'Xw': (R_wc.T @ (point_3d.reshape(3,1) - T_wc)).flatten()[0],
                    'Yw': (R_wc.T @ (point_3d.reshape(3,1) - T_wc)).flatten()[1],
                    'Zw': (R_wc.T @ (point_3d.reshape(3,1) - T_wc)).flatten()[2]
                })
        
        calculated_df = pd.DataFrame(world_coords)
        
        # 3. Align and calculate errors
        print("Calculating alignment and errors...")
        merged_df = pd.merge(calculated_df, truth_df, on='marker_id')
        if merged_df.empty:
            raise ValueError("No common markers between calculated and truth data")
        
        calc_points = merged_df[['Xw', 'Yw', 'Zw']].values
        truth_points = merged_df[['Xw_truth', 'Yw_truth', 'Zw_truth']].values
        
        # Calculate optimal shift
        centroid_diff = np.mean(calc_points, axis=0) - np.mean(truth_points, axis=0)
        aligned_truth = truth_points + centroid_diff
        
        # Calculate errors
        errors = calc_points - aligned_truth
        total_errors = np.linalg.norm(errors, axis=1)
        
        # 4. Create visualizations
        print("Creating visualizations...")
        _ = create_3d_comparison_plot(calc_points, aligned_truth, total_errors, output_3d)
        _ = create_error_analysis_plot(merged_df['marker_id'].values, errors, output_error)
        
        print(f"Saved 3D plot to: {output_3d}")
        print(f"Saved error analysis plot to: {output_error}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return

if __name__ == "__main__":
    print("Starting pose comparison analysis...")
    main()
    print("Analysis complete.")