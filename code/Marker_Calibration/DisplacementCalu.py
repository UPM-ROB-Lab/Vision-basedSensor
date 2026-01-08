#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os
import chardet
import traceback
from typing import Dict, Tuple, Optional

# ------------------------------
# Configuration (Modify these or move to config file)
# ------------------------------
CONFIG = {
    'marker': {
        'diameter_mm': 2.0,  # Circle marker diameter
        'warmup_frames': 100  # Camera warmup frames
    },
    'paths': {
        'data_dir': "Results/data",
        'output_dir': "Results/data/results",
        'plots_dir': "Results/data/results/Displacement_Analysis_Plots"
    },
    'column_mapping': {
        'Cx': 'u',            # X-coordinate column
        'Cy': 'v',            # Y-coordinate column 
        'major_axis': 'major_axis'  # Marker major axis column
    }
}

def load_camera_parameters(intrinsic_path: str, extrinsic_path: str) -> Tuple:
    """Load camera parameters from Excel files with validation."""
    try:
        # Load intrinsic parameters
        intrinsic_df = pd.read_excel(intrinsic_path)
        params = intrinsic_df.set_index('Parameter')['Value']
        camera_matrix = np.array([
            [params['fx'], params.get('skew', 0), params['cx']],
            [0, params['fy'], params['cy']],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Load distortion coefficients
        dist_keys = ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6']
        dist_coeffs = np.array([params.get(k, 0) for k in dist_keys], dtype=np.float32)
        
        # Load extrinsic parameters
        extrinsic_df = pd.read_excel(extrinsic_path)
        ext_params = extrinsic_df.set_index('Parameter')['Value']
        R_world_to_cam = np.array([
            [ext_params[f'R_wc_{i}{j}'] for j in range(1,4)] 
            for i in range(1,4)
        ], dtype=np.float32)
        
        T_world_to_cam = np.array([
            ext_params['Tx_wc'], 
            ext_params['Ty_wc'], 
            ext_params['Tz_wc']
        ], dtype=np.float32).reshape(3, 1)
        
        return camera_matrix, dist_coeffs, R_world_to_cam, T_world_to_cam
        
    except Exception as e:
        raise ValueError(f"Parameter loading failed: {str(e)}")

def load_marker_data(filepath: str) -> pd.DataFrame:
    """Load and validate marker data with robust parsing."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Detect file encoding
    with open(filepath, 'rb') as f:
        encoding = chardet.detect(f.read(30000))['encoding'] or 'utf-8'
    
    try:
        df = pd.read_csv(
            filepath, 
            sep=r'\s+|,|\t', 
            encoding=encoding, 
            engine='python', 
            skipinitialspace=True
        )
    except Exception as e:
        raise ValueError(f"CSV parsing failed: {str(e)}")
    
    # Clean and validate columns
    df.columns = [col.strip() for col in df.columns]
    required_cols = set(CONFIG['column_mapping'].keys()) | {'frameno', 'row', 'col'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df.rename(columns=CONFIG['column_mapping'])

def calculate_displacement(initial: Dict, current: Dict, 
                          camera_matrix: np.ndarray,
                          R_wc: np.ndarray, 
                          T_wc: np.ndarray) -> Optional[Tuple]:
    """Calculate 3D displacement between two marker states."""
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    f_avg = (fx + fy) / 2
    
    u1, v1, x1 = initial['u'], initial['v'], initial['major_axis']
    u2, v2, x2 = current['u'], current['v'], current['major_axis']
    
    # Validate inputs
    if x1 < 1e-6 or abs((x2 - x1)/x1 + 1) < 1e-6:
        return None
    
    # Calculate initial depth
    R1 = np.sqrt((u1 - cx)**2 + (v1 - cy)**2)
    d1_effective = (CONFIG['marker']['diameter_mm'] / f_avg) * np.sqrt(R1**2 + f_avg**2)
    h1 = f_avg * (d1_effective / x1)
    
    # Calculate displacement
    R2 = np.sqrt((u2 - cx)**2 + (v2 - cy)**2)
    alpha = (x2 - x1) / x1
    term_ratio = np.sqrt((R1**2 + f_avg**2) / (R2**2 + f_avg**2))
    delta_h = h1 * (1 - (term_ratio / (alpha + 1)))
    
    # Convert to 3D coordinates
    def pixel_to_world(u, v, h):
        Xc = h * (u - cx) / fx
        Yc = h * (v - cy) / fy
        P_cam = np.array([Xc, Yc, h]).reshape(3, 1)
        return (R_wc.T @ (P_cam - T_wc)).flatten()
    
    P_world1 = pixel_to_world(u1, v1, h1)
    P_world2 = pixel_to_world(u2, v2, h1 - delta_h)
    
    return P_world2, P_world1, (P_world2 - P_world1)

def analyze_displacement(results_df: pd.DataFrame, output_dir: str) -> None:
    """Generate frame-to-frame displacement analysis plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate frame differences
    plot_df = results_df.sort_values(['marker_id', 'frameno']).copy()
    diffs = plot_df.groupby('marker_id')[['Xw', 'Yw', 'Zw']].diff()
    diffs.columns = ['dX_frame', 'dY_frame', 'dZ_frame']
    plot_df = pd.concat([plot_df, diffs], axis=1)
    
    plot_df['displacement'] = np.sqrt(
        plot_df['dX_frame']**2 + 
        plot_df['dY_frame']**2 + 
        plot_df['dZ_frame']**2
    )
    
    # Generate plots
    for marker_id, data in plot_df.groupby('marker_id'):
        if data['displacement'].notna().any():
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data['frameno'], data['displacement'], 'o-', markersize=4)
            ax.set(
                title=f'Frame Displacement - Marker {marker_id}',
                xlabel='Frame Number',
                ylabel='3D Displacement (mm)',
                ylim=(0, None)
            )
            ax.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f'marker_{marker_id}_displacement.png'), 
                dpi=150
            )
            plt.close()

def run_analysis():
    """Main analysis pipeline."""
    try:
        # Setup paths
        data_dir = CONFIG['paths']['data_dir']
        input_csv = os.path.join(data_dir, 'marker_locations_0.csv')
        intrinsic_path = os.path.join(data_dir, 'PreprocessPara', 'IntrinsicParameters.xlsx')
        extrinsic_path = os.path.join(data_dir, 'PreprocessPara', 'ExtrinsicParameters.xlsx')
        
        # Load data
        camera_matrix, dist_coeffs, R_wc, T_wc = load_camera_parameters(
            intrinsic_path, extrinsic_path
        )
        df = load_marker_data(input_csv)
        
        # Undistort points
        points = df[['u', 'v']].values.reshape(-1, 1, 2)
        df[['u', 'v']] = cv2.undistortPoints(
            points, camera_matrix, dist_coeffs, None, camera_matrix
        ).reshape(-1, 2)
        
        # Process frames
        initial_frame = df[df['frameno'] == df['frameno'].min()].copy()
        initial_frame['marker_id'] = range(1, len(initial_frame)+1)
        initial_states = initial_frame.set_index('marker_id').to_dict('index')
        
        results = []
        for frame in df['frameno'].unique():
            current = df[df['frameno'] == frame].copy()
            current['marker_id'] = range(1, len(current)+1)
            
            for _, row in current.iterrows():
                if row['marker_id'] in initial_states:
                    world_pos = calculate_displacement(
                        initial_states[row['marker_id']],
                        row.to_dict(),
                        camera_matrix,
                        R_wc,
                        T_wc
                    )
                    if world_pos:
                        results.append({
                            'frameno': frame,
                            'marker_id': row['marker_id'],
                            'Xw': world_pos[0][0],
                            'Yw': world_pos[0][1],
                            'Zw': world_pos[0][2],
                            **{f'd{axis}': world_pos[2][i] for i, axis in enumerate('XYZ')}
                        })
        
        # Save and analyze results
        results_df = pd.DataFrame(results)
        output_path = os.path.join(CONFIG['paths']['output_dir'], 'marker_3d_coordinates.xlsx')
        results_df.to_excel(output_path, index=False)
        
        analyze_displacement(
            results_df, 
            CONFIG['paths']['plots_dir']
        )
        
        print(f"Analysis completed successfully. Results saved to {output_path}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    run_analysis()