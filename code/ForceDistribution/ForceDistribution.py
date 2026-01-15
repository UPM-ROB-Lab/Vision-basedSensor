import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from io import StringIO
import traceback

# -----------------------------------------------------------------------------
# Configuration Parameters
# -----------------------------------------------------------------------------
# Visualization Settings
DEVIATION_SCALE_FACTOR = 1.0
INITIAL_PLOT_MODE = 'plane' 
FIT_TILTED_PLANE = True  
VIEW_ELEV = 20      
VIEW_AZIM = 45   

# File Paths
DATA_DIR = "data"
INITIAL_STATE_FILE = "initial4.txt"
TILTED_STATE_FILE = "40.txt"

# -----------------------------------------------------------------------------
# Reference Data (Embedded)
# -----------------------------------------------------------------------------
# Initial theoretical coordinates (P_start)
MARKER_REF_DATA = """
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

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def set_axes_equal(ax):
    """Sets equal aspect ratio for 3D plots to prevent geometric distortion."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def load_experimental_data(file_path):
    """Parses experimental TXT files into a standardized DataFrame."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Locate data start (handling potential header variations)
        header_keywords = ["MarkerID", "marker_id"]
        data_start = -1
        for keyword in header_keywords:
            pos = content.find(keyword)
            if pos != -1:
                data_start = pos
                break
        
        if data_start == -1:
            raise ValueError("Header not found in file.")

        df = pd.read_csv(StringIO(content[data_start:]), sep='\s+')
        
        # Standardize column names
        col_map = {'marker_id': 'MarkerID'}
        df = df.rename(columns=col_map)
        return df.set_index('MarkerID')
    
    except Exception as e:
        raise IOError(f"Failed to load {file_path}: {e}")

def fit_plane_least_squares(ax, X, Y, Z, color='orange', label='Fitted Plane'):
    """Fits and plots a plane using least squares method."""
    # Formulate matrix A = [X, Y, 1]
    A = np.vstack([X, Y, np.ones(len(X))]).T
    try:
        # Solve Z = aX + bY + c
        coeff, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
        a, b, c = coeff
    except np.linalg.LinAlgError:
        print(f"Warning: Plane fitting failed for {label}")
        return None, None

    # Generate grid for plotting
    x_grid = np.linspace(X.min(), X.max(), 10)
    y_grid = np.linspace(Y.min(), Y.max(), 10)
    XX, YY = np.meshgrid(x_grid, y_grid)
    ZZ = a * XX + b * YY + c

    ax.plot_surface(XX, YY, ZZ, color=color, alpha=0.3, linewidth=0)
    
    # Calculate tilt angle
    tilt_deg = np.degrees(np.arctan(np.sqrt(a**2 + b**2)))
    print(f"-> Plane Fit ({label}): Tilt Angle = {tilt_deg:.2f} degrees")
    
    return Patch(color=color, alpha=0.3), label

# -----------------------------------------------------------------------------
# Data Processing Logic
# -----------------------------------------------------------------------------

def process_marker_data():
    """Loads data and calculates deviations (without outlier filtering)."""
    print("--- Step 1: Data Loading and Processing ---")
    
    # 1. Load Reference Data
    df_ref = pd.read_csv(StringIO(MARKER_REF_DATA), sep='\s+', header=None, 
                         names=['id', 'X', 'Y', 'Z']).set_index('id')
    
    # 2. Load Experimental Data
    path_init = os.path.join(DATA_DIR, INITIAL_STATE_FILE)
    path_tilt = os.path.join(DATA_DIR, TILTED_STATE_FILE)
    
    df_vert = load_experimental_data(path_init)
    df_tilt = load_experimental_data(path_tilt)
    
    # 3. Find Common Markers
    common_ids = df_vert.index.intersection(df_tilt.index).intersection(df_ref.index)
    if common_ids.empty:
        raise ValueError("No common markers found across datasets.")
    
    # 4. Calculate Deviations
    df = pd.DataFrame(index=common_ids)
    
    # Store original reference coordinates
    df['X_ref'] = df_ref.loc[common_ids, 'X']
    df['Y_ref'] = df_ref.loc[common_ids, 'Y']
    df['Z_ref'] = df_ref.loc[common_ids, 'Z']
    
    # Calculate displacement vectors (Tilt - Vertical)
    d_vert = df_vert.loc[common_ids, ['X_end', 'Y_end', 'Z_end']].values - \
             df_vert.loc[common_ids, ['X_start', 'Y_start', 'Z_start']].values
             
    d_tilt = df_tilt.loc[common_ids, ['X_end', 'Y_end', 'Z_end']].values - \
             df_tilt.loc[common_ids, ['X_start', 'Y_start', 'Z_start']].values
    
    deviation = d_tilt - d_vert
    df['dX'], df['dY'], df['dZ'] = deviation[:, 0], deviation[:, 1], deviation[:, 2]
    
    print(f"-> Data processing complete. Total markers analyzed: {len(df)}")
    
    return df

# -----------------------------------------------------------------------------
# Visualization Logic
# -----------------------------------------------------------------------------

def visualize_deviations(df):
    """Generates the 3D vector plot."""
    print("--- Step 2: Generating Visualization ---")
    
    # Prepare Coordinates
    X_start = df['X_ref']
    Y_start = df['Y_ref']
    # Determine Z based on mode
    Z_start = df['Z_ref'] if INITIAL_PLOT_MODE == 'shell' else np.zeros_like(df['Z_ref'])
    
    # Apply Scaling Factor
    dX_scaled = df['dX'] * DEVIATION_SCALE_FACTOR
    dY_scaled = df['dY'] * DEVIATION_SCALE_FACTOR
    dZ_scaled = df['dZ'] * DEVIATION_SCALE_FACTOR
    
    # Calculate End Points for Plotting
    X_end = X_start + dX_scaled
    Y_end = Y_start + dY_scaled
    Z_end = Z_start + dZ_scaled
    
    # Setup Plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    legend_elements = []
    
    # 1. Plot Plane (Optional)
    if FIT_TILTED_PLANE:
        handle, label = fit_plane_least_squares(ax, X_end, Y_end, Z_end)
        if handle:
            legend_elements.append((handle, label))
            
    # 2. Plot Initial Positions
    sc = ax.scatter(X_start, Y_start, Z_start, c='blue', s=50, alpha=0.8, edgecolors='k')
    legend_elements.append((sc, 'Initial Position ($P_{ref}$)'))
    
    # 3. Plot Vectors
    ax.quiver(X_start, Y_start, Z_start, dX_scaled, dY_scaled, dZ_scaled,
              color='red', arrow_length_ratio=0.2, linewidth=1.5, alpha=0.8)
    # Create a proxy artist for the quiver legend
    legend_elements.append((Patch(color='red'), f'Deviation Vector (x{DEVIATION_SCALE_FACTOR})'))
    
    # 4. Plot End Points
    sc_end = ax.scatter(X_end, Y_end, Z_end, c='red', marker='s', s=30, alpha=0.6)
    
    # 5. Plot Average Vector
    avg_vec = np.mean([dX_scaled, dY_scaled, dZ_scaled], axis=1)
    center = [X_start.mean(), Y_start.mean(), Z_start.mean()]
    ax.quiver(center[0], center[1], center[2], avg_vec[0], avg_vec[1], avg_vec[2],
              color='green', linewidth=4, arrow_length_ratio=0.2)
    legend_elements.append((Patch(color='green'), 'Mean Deviation Vector'))

    # 6. Labels and Formatting
    # Note: Z-offset (+0.5) is manually applied for label visibility
    for idx, x, y, z in zip(df.index, X_start, Y_start, Z_start):
        ax.text(x, y, z + 0.5, str(int(idx)), color='purple', fontsize=8, weight='bold')

    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_zlabel('Z (mm)', fontsize=12)
    
    mean_mag = np.linalg.norm(df[['dX', 'dY', 'dZ']].values, axis=1).mean()
    ax.set_title(f'3D Deviation Analysis ({INITIAL_PLOT_MODE} view)\n'
                 f'Mean Magnitude: {mean_mag:.4f} mm', fontsize=14)
    
    # Unpack legend elements
    handles, labels = zip(*legend_elements)
    ax.legend(handles, labels, loc='upper right', fontsize=10)
    
    set_axes_equal(ax)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        print("=== Starting 3D Deviation Analysis ===")
        df_analysis = process_marker_data()
        visualize_deviations(df_analysis)
        print("=== Analysis Complete ===")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        traceback.print_exc()