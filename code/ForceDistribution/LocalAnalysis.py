import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------------------------------------------------------
# Configuration Parameters
# -----------------------------------------------------------------------------
# List of marker IDs to analyze
TARGET_MARKERS = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

# Frame ranges for averaging (inclusive)
START_FRAME_RANGE = (1, 30)
END_FRAME_RANGE = (120, 150)

# File paths
DATA_DIR = "data"
INPUT_FILE = "marker_3d_coordinates.xlsx"
OUTPUT_FILE = "Averaged_Displacement.png"

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def set_axes_equal(ax):
    """
    Sets equal aspect ratio for 3D plots to prevent geometric distortion.
    
    Args:
        ax: Matplotlib 3D axis object.
    """
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def load_and_filter_data(filepath, markers):
    """Loads Excel data and filters by specific marker IDs."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    df = pd.read_excel(filepath)
    df_filtered = df[df['marker_id'].isin(markers)]
    
    if df_filtered.empty:
        raise ValueError("No data found for the specified target markers.")
    return df_filtered

def calculate_average_coordinates(df, frame_range, suffix):
    """Calculates mean X, Y, Z coordinates for markers within a frame range."""
    mask = (df['frameno'] >= frame_range[0]) & (df['frameno'] <= frame_range[1])
    data_subset = df[mask]
    
    avg_df = data_subset.groupby('marker_id')[['Xw', 'Yw', 'Zw']].mean().reset_index()
    avg_df.columns = ['marker_id', f'X_{suffix}', f'Y_{suffix}', f'Z_{suffix}']
    return avg_df

# -----------------------------------------------------------------------------
# Main Analysis Function
# -----------------------------------------------------------------------------

def analyze_displacement():
    """Main routine to process data and generate the 3D plot."""
    input_path = os.path.join(DATA_DIR, INPUT_FILE)
    output_path = os.path.join(DATA_DIR, OUTPUT_FILE)
    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        # 1. Load Data
        df = load_and_filter_data(input_path, TARGET_MARKERS)

        # 2. Calculate Averages
        start_avg = calculate_average_coordinates(df, START_FRAME_RANGE, 'start')
        end_avg = calculate_average_coordinates(df, END_FRAME_RANGE, 'end')

        # 3. Merge and Calculate Vectors
        merged = pd.merge(start_avg, end_avg, on='marker_id', how='inner')
        
        if merged.empty:
            print("Error: No common markers found between ranges.")
            return

        # Calculate displacement components
        merged['dX'] = merged['X_end'] - merged['X_start']
        merged['dY'] = merged['Y_end'] - merged['Y_start']
        merged['dZ'] = merged['Z_end'] - merged['Z_start']

        # Calculate magnitude (Euclidean distance)
        magnitudes = np.linalg.norm(merged[['dX', 'dY', 'dZ']].values, axis=1)
        print(f"Analysis complete. Mean displacement: {np.mean(magnitudes):.4f} mm")

        # 4. Visualization
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot Start Positions
        ax.scatter(
            merged['X_start'], merged['Y_start'], merged['Z_start'],
            c='blue', marker='o', s=80, edgecolors='k', alpha=0.6,
            label='Start Position (Avg)'
        )

        # Plot End Positions
        ax.scatter(
            merged['X_end'], merged['Y_end'], merged['Z_end'],
            c='red', marker='P', s=100, alpha=0.8,
            label='End Position (Avg)'
        )

        # Plot Displacement Vectors
        ax.quiver(
            merged['X_start'], merged['Y_start'], merged['Z_start'],
            merged['dX'], merged['dY'], merged['dZ'],
            color='green', arrow_length_ratio=0.1, linewidth=2.0,
            label='Displacement Vector', alpha=0.8
        )

        # Add Labels
        for _, row in merged.iterrows():
            # Note: A manual offset (+1) is applied to Z for label visibility.
            ax.text(
                row['X_start'], row['Y_start'], row['Z_start'] + 1, 
                f"M{int(row['marker_id'])}", 
                color='purple', fontsize=9, weight='bold'
            )

        # Formatting
        ax.set_xlabel('World X (mm)', fontsize=12)
        ax.set_ylabel('World Y (mm)', fontsize=12)
        ax.set_zlabel('World Z (mm)', fontsize=12)
        ax.set_title('Averaged 3D Marker Displacement', fontsize=14, weight='bold')
        ax.legend(loc='best', fontsize=10)
        
        set_axes_equal(ax)
        plt.tight_layout()

        # Save Figure
        plt.savefig(output_path, dpi=400, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    analyze_displacement()