import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from io import StringIO
import traceback
import re

# ------------------------------
# Visualization Configuration
# ------------------------------
DEVIATION_SCALE_FACTOR = 1  # Displacement scaling factor
INITIAL_PLOT_MODE = 'plane'  # 'shell' (use original Z coordinates) or 'plane' (set Z axis to 0)
FIT_TILTED_PLANE = True  # Whether to fit a plane to the tilted final positions
OUTLIER_Z_THRESHOLD = 3 # New: Z-score threshold for outlier removal (e.g., 3.0)
# ------------------------------
# File Path Configuration
# ------------------------------
BASE_DIR = os.path.join("data")
INITIAL_TXT_PATH = os.path.join(BASE_DIR, 'initial4.txt')  # Initial state file
TILTED_TXT_PATH = os.path.join(BASE_DIR, '4no.txt')   # Tilted state file

# ------------------------------
# Marker Coordinates Data - Used to get P_start
marker_data_str = """
1	0.00	0.00	0.00
2	-3.02	1.74	0.23
3	0.00	3.49	0.23
4	3.02	1.74	0.23
5	3.02	-1.74	0.23
6	0.00	-3.49	0.23
7	-3.02	-1.74	0.23
8	-3.46	5.99	0.90
9	0.00	6.92	0.90
10	3.46	5.99	0.90
11	5.99	3.46	0.90
12	6.92	0.00	0.90
13	5.99	-3.46	0.90
14	3.46	-5.99	0.90
15	0.00	-6.92	0.90
16	-3.46	-5.99	0.90
17	-5.99	-3.46	0.90
18	-6.92	0.00	0.90
19	-5.99	3.46	0.90
20	-6.58	7.84	2.01
21	-3.50	9.61	2.01
22	0.00	10.23	2.01
23	3.50	9.61	2.01
24	6.58	7.84	2.01
25	8.86	5.11	2.01
26	10.07	1.78	2.01
27	10.07	-1.78	2.01
28	8.86	-5.11	2.01
29	6.58	-7.84	2.01
30	3.50	-9.61	2.01
31	0.00	-10.23	2.01
32	-3.50	-9.61	2.01
33	-6.58	-7.84	2.01
34	-8.86	-5.11	2.01
35	-10.07	-1.78	2.01
36	-10.07	1.78	2.01
37	-8.86	5.11	2.01
38	-9.45	9.45	3.55
39	-6.69	11.58	3.55
40	-3.46	12.92	3.55
41	0.00	13.37	3.55
42	3.46	12.92	3.55
43	6.69	11.58	3.55
44	9.45	9.45	3.55
45	11.58	6.69	3.55
46	12.92	3.46	3.55
47	13.37	0.00	3.55
48	12.92	-3.46	3.55
49	11.58	-6.69	3.55
50	9.45	-9.45	3.55
51	6.69	-11.58	3.55
52	3.46	-12.92	3.55
53	0.00	-13.37	3.55
54	-3.46	-12.92	3.55
55	-6.69	-11.58	3.55
56	-9.45	-9.45	3.55
57	-11.58	-6.69	3.55
58	-12.92	-3.46	3.55
59	-13.37	0.00	3.55
60	-12.92	3.46	3.55
61	-11.58	6.69	3.55
62	0.00	16.29	5.47
63	16.29	0.00	5.47
64	0.00	-16.29	5.47
65	-16.29	0.00	5.47
"""

# =============================================================================
# Core Helper Functions
# =============================================================================
def set_axes_equal(ax):
    """Ensure 3D plot axes have equal scale to prevent geometric distortion."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def load_avg_coordinates_from_txt(file_path):
    """
    Load marker_id and average coordinates from TXT file, adapted to your file format
    """
    try:
        # Read entire file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find start of data table
        data_start = content.find("MarkerID")
        if data_start == -1:
            # Try alternative column names
            data_start = content.find("marker_id")
            if data_start == -1:
                raise ValueError("Cannot find data table header (MarkerID or marker_id)")
        
        # Read data section
        df = pd.read_csv(StringIO(content[data_start:]), sep='\s+')
        
        # Standardize column names
        column_mapping = {
            'marker_id': 'MarkerID',
            'X_start': 'X_start',
            'Y_start': 'Y_start',
            'Z_start': 'Z_start',
            'X_end': 'X_end',
            'Y_end': 'Y_end',
            'Z_end': 'Z_end'
        }
        
        # Rename columns
        df = df.rename(columns={col: column_mapping.get(col, col) for col in df.columns})
        
        # Check required columns
        required_cols = ['MarkerID', 'X_start', 'Y_start', 'Z_start', 'X_end', 'Y_end', 'Z_end']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"File missing required columns: {missing_cols}")
            
        return df.set_index('MarkerID')
    
    except Exception as e:
        raise Exception(f"Error reading file {file_path}: {str(e)}")

def fit_and_plot_plane(ax, X, Y, Z, color, label, alpha=0.4):
    """
    Perform least squares plane fitting on given (X, Y, Z) coordinates and plot the fitted plane.
    Returns a proxy handle for legend.
    """
    # Construct matrix A: [X, Y, 1]
    A = np.vstack([X, Y, np.ones(len(X))]).T
    
    # Solve for coefficients [a, b, c] using least squares
    try:
        coeff, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
        a, b, c = coeff
    except np.linalg.LinAlgError:
        print(f"-> Warning: Plane fitting failed for {label}")
        return None, None
    
    # Plot fitted plane
    X_range = np.linspace(X.min(), X.max(), 10)
    Y_range = np.linspace(Y.min(), Y.max(), 10)
    XX, YY = np.meshgrid(X_range, Y_range)
    ZZ = a * XX + b * YY + c
    
    ax.plot_surface(XX, YY, ZZ, color=color, alpha=alpha, linewidth=0)
    
    # Calculate tilt angle
    gradient_magnitude = np.sqrt(a**2 + b**2)
    tilt_angle_deg = np.degrees(np.arctan(gradient_magnitude))
    
    print(f"-> Plane fit for {label}: Z = {a:.4f}X + {b:.4f}Y + {c:.4f} (Tilt angle: {tilt_angle_deg:.2f} degrees)")
    
    # Return legend handle
    from matplotlib.patches import Patch
    return Patch(color=color, alpha=alpha), label

# =============================================================================
# Core Data Processing Functions
# =============================================================================
def prepare_data():
    """Load all data and calculate deviation vectors, including outlier filtering."""
    print("\n--- Step 1: Loading and preparing deviation data ---")
    
    # 1. Load reference marker data
    df_start_all = pd.read_csv(StringIO(marker_data_str), sep='\s+', header=None, 
                             names=['id', 'X', 'Y', 'Z']).set_index('id')
    
    # 2. Load experimental data files
    try:
        print(f"Loading file: {os.path.basename(INITIAL_TXT_PATH)}")
        df_vert = load_avg_coordinates_from_txt(INITIAL_TXT_PATH)
        
        print(f"Loading file: {os.path.basename(TILTED_TXT_PATH)}")
        df_tilt = load_avg_coordinates_from_txt(TILTED_TXT_PATH)
    except Exception as e:
        print(f"\n[Error] File loading failed:")
        print(f"Initial state file: {INITIAL_TXT_PATH}")
        print(f"Tilted state file: {TILTED_TXT_PATH}")
        raise
    
    # 3. Find common marker points
    common_ids = df_vert.index.intersection(df_tilt.index).intersection(df_start_all.index)
    if common_ids.empty:
        raise Exception("Error: No common markers found across all data sources")
    
    print(f"Found {len(common_ids)} common markers initially.")
    
    # 4. Prepare analysis dataframe
    df_analysis = pd.DataFrame(index=common_ids)
    
    # Store original coordinates
    df_analysis['X_start_orig'] = df_start_all.loc[common_ids, 'X']
    df_analysis['Y_start_orig'] = df_start_all.loc[common_ids, 'Y']
    df_analysis['Z_start_orig'] = df_start_all.loc[common_ids, 'Z']
    
    # Calculate vertical displacement
    df_analysis['dX_vert'] = df_vert.loc[common_ids, 'X_end'] - df_vert.loc[common_ids, 'X_start']
    df_analysis['dY_vert'] = df_vert.loc[common_ids, 'Y_end'] - df_vert.loc[common_ids, 'Y_start']
    df_analysis['dZ_vert'] = df_vert.loc[common_ids, 'Z_end'] - df_vert.loc[common_ids, 'Z_start']
    
    # Calculate tilted state displacement
    df_analysis['dX_tilt'] = df_tilt.loc[common_ids, 'X_end'] - df_tilt.loc[common_ids, 'X_start']
    df_analysis['dY_tilt'] = df_tilt.loc[common_ids, 'Y_end'] - df_tilt.loc[common_ids, 'Y_start']
    df_analysis['dZ_tilt'] = df_tilt.loc[common_ids, 'Z_end'] - df_tilt.loc[common_ids, 'Z_start']
    
    # Calculate deviation displacement (tilted - vertical)
    df_analysis['dX_dev'] = df_analysis['dX_tilt'] - df_analysis['dX_vert']
    df_analysis['dY_dev'] = df_analysis['dY_tilt'] - df_analysis['dY_vert']
    df_analysis['dZ_dev'] = df_analysis['dZ_tilt'] - df_analysis['dZ_vert']
    
    # 5. Outlier Filtering based on Deviation Magnitude (New Step)
    
    # Calculate magnitude of the deviation vector
    deviation_vectors = df_analysis[['dX_dev', 'dY_dev', 'dZ_dev']].values
    magnitudes = np.linalg.norm(deviation_vectors, axis=1)
    df_analysis['Magnitude'] = magnitudes
    
    # Calculate mean and standard deviation of magnitudes
    mu_L = magnitudes.mean()
    sigma_L = magnitudes.std()
    
    # Calculate Z-score for each magnitude
    z_scores = np.abs(magnitudes - mu_L) / sigma_L
    
    # Filter out outliers
    outlier_mask = z_scores > OUTLIER_Z_THRESHOLD
    outlier_ids = df_analysis.index[outlier_mask]
    
    df_analysis_filtered = df_analysis[~outlier_mask].copy()
    
    print(f"-> Deviation Magnitude Mean: {mu_L:.4f}, Std Dev: {sigma_L:.4f}")
    print(f"-> Outlier Z-Threshold set to: {OUTLIER_Z_THRESHOLD}")
    print(f"-> Removed {len(outlier_ids)} outliers (IDs: {list(outlier_ids)})")
    print(f"-> Remaining markers for analysis: {len(df_analysis_filtered)}")
    
    # 6. Apply scaling factor to filtered data
    df_analysis_filtered['dX_dev_scaled'] = df_analysis_filtered['dX_dev'] * DEVIATION_SCALE_FACTOR
    df_analysis_filtered['dY_dev_scaled'] = df_analysis_filtered['dY_dev'] * DEVIATION_SCALE_FACTOR
    df_analysis_filtered['dZ_dev_scaled'] = df_analysis_filtered['dZ_dev'] * DEVIATION_SCALE_FACTOR
    
    # Calculate average deviation magnitude for the filtered set
    mean_deviation_filtered = np.mean(np.linalg.norm(df_analysis_filtered[['dX_dev', 'dY_dev', 'dZ_dev']].values, axis=1))
    print(f"Average deviation magnitude (Filtered): {mean_deviation_filtered:.4f} mm")
    
    return df_analysis_filtered

# =============================================================================
# Plotting Functions
# =============================================================================
def draw_plot(df_analysis, mode):
    """Draw 3D deviation vector plot according to mode"""
    
    # 1. Determine starting coordinates
    if mode == 'plane':
        X_start = df_analysis['X_start_orig']
        Y_start = df_analysis['Y_start_orig']
        Z_start = np.zeros_like(df_analysis['Z_start_orig'])  # Z=0 plane projection
        title_suffix = " (plane projection, Z=0)"
    elif mode == 'shell':
        X_start = df_analysis['X_start_orig']
        Y_start = df_analysis['Y_start_orig']
        Z_start = df_analysis['Z_start_orig']  # Keep original Z coordinates
        title_suffix = " (shell surface)"
    else:
        raise ValueError("Invalid plot mode, must be 'shell' or 'plane'")
    
    # 2. Calculate deviation end coordinates
    X_plot = X_start + df_analysis['dX_dev_scaled']
    Y_plot = Y_start + df_analysis['dY_dev_scaled']
    Z_plot = Z_start + df_analysis['dZ_dev_scaled']
    
    # 3. Calculate overall average deviation
    dX_dev_avg = df_analysis['dX_dev_scaled'].mean()
    dY_dev_avg = df_analysis['dY_dev_scaled'].mean()
    dZ_dev_avg = df_analysis['dZ_dev_scaled'].mean()
    
    # 4. Create 3D plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Handles and labels for legend
    plot_handles = []
    plot_labels = []
    
    print("\n--- Step 2: Drawing 3D deviation plot ---")
    
    # 5. Fit and plot plane (if enabled)
    if FIT_TILTED_PLANE:
        print("-> Fitting tilted plane...")
        plane_handle, plane_label = fit_and_plot_plane(
            ax, X_plot, Y_plot, Z_plot, 
            color='orange', 
            label='Fitted tilted plane',
            alpha=0.3
        )
        if plane_handle:
            plot_handles.append(plane_handle)
            plot_labels.append(plane_label)
    
    # 6. Plot markers and vectors
    
    # 6.1 Initial positions
    scatter_start = ax.scatter(
        X_start, Y_start, Z_start,
        c='blue', marker='o', s=50, edgecolors='k', alpha=0.9,
        label='Initial position ($P_{start}$)'
    )
    plot_handles.append(scatter_start)
    plot_labels.append('Initial position ($P_{start}$)')
    
    # 6.2 Deviation vectors (red arrows)
    quiver_individual = ax.quiver(
        X_start, Y_start, Z_start,
        df_analysis['dX_dev_scaled'], df_analysis['dY_dev_scaled'], df_analysis['dZ_dev_scaled'],
        color='red', arrow_length_ratio=0.2, linewidth=1.5, alpha=0.8,
        label=f'Individual deviation (scaled x{DEVIATION_SCALE_FACTOR:.1f})'
    )
    plot_handles.append(quiver_individual)
    plot_labels.append(f'Individual deviation (x{DEVIATION_SCALE_FACTOR:.1f})')
    
    # 6.3 Deviation end points
    scatter_end = ax.scatter(
        X_plot, Y_plot, Z_plot,
        c='red', marker='s', s=80, edgecolors='k', linewidths=1,
        label='Deviation end position'
    )
    plot_handles.append(scatter_end)
    plot_labels.append('Deviation end position')
    
    # 6.4 Average deviation vector (large green arrow)
    X_start_avg = X_start.mean()
    Y_start_avg = Y_start.mean()
    Z_start_avg = Z_start.mean()
    
    quiver_avg = ax.quiver(
        X_start_avg, Y_start_avg, Z_start_avg,
        dX_dev_avg, dY_dev_avg, dZ_dev_avg,
        color='green', arrow_length_ratio=0.2, linewidth=4, alpha=1.0,
        label=f'Overall average deviation (scaled x{DEVIATION_SCALE_FACTOR:.1f})',
        zorder=10
    )
    plot_handles.append(quiver_avg)
    plot_labels.append(f'Overall average deviation (x{DEVIATION_SCALE_FACTOR:.1f})')
    
    # 6.5 Marker IDs
    for marker_id, x, y, z in zip(df_analysis.index, X_start, Y_start, Z_start):
        ax.text(x, y, z + 0.5, str(marker_id), color='purple', fontsize=9, weight='bold')
    
    # 7. Set plot style
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_zlabel('Z (mm)', fontsize=12)
    
    mean_deviation = np.mean(np.linalg.norm(df_analysis[['dX_dev', 'dY_dev', 'dZ_dev']].values, axis=1))
    ax.set_title(
        f'Displacement deviation due to tilt{title_suffix}\n(Average deviation: {mean_deviation:.4f} mm, Scale factor: x{DEVIATION_SCALE_FACTOR:.1f})',
        fontsize=14, pad=20
    )
    
    ax.legend(plot_handles, plot_labels, loc='upper right', fontsize=10)
    set_axes_equal(ax)
    ax.view_init(elev=20, azim=45)
    
    print("--- Step 3: Displaying 3D plot ---")
    plt.tight_layout()
    plt.show()

# =============================================================================
# Main Program
# =============================================================================
def plot_deviation_vector_controlled():
    """Main control function"""
    print("\n=== 3D Deviation Vector Analysis Program ===")
    print(f"Plot mode: {INITIAL_PLOT_MODE}")
    print(f"Deviation scale factor: {DEVIATION_SCALE_FACTOR}")
    print(f"Fit tilted plane: {'Yes' if FIT_TILTED_PLANE else 'No'}")
    
    try:
        # Load and prepare data
        df_analysis = prepare_data()
        
        # Draw plot
        draw_plot(df_analysis, INITIAL_PLOT_MODE)
        
    except Exception as e:
        print(f"\n[Critical Error] Error during analysis: {str(e)}")
        traceback.print_exc()
    finally:
        print("\nAnalysis complete.")

if __name__ == "__main__":
    plot_deviation_vector_controlled()