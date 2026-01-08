import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import traceback

# ------------------------------
# Configuration Parameters
# ------------------------------
START_FRAME = 1
END_FRAME = 120
DPI = 400  # High resolution for publication quality
MARKER_SIZE = 80  # Base marker size
FONT_SIZE = 12  # Base font size

# =============================================================================
# Core Visualization Functions
# =============================================================================
def set_axes_equal(ax):
    """Set equal scale for 3D axes"""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def load_and_process_data(input_path):
    """Load and validate marker coordinate data"""
    print("Loading and processing data...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_excel(input_path)
    start_data = df[df['frameno'] == START_FRAME].set_index('marker_id')
    end_data = df[df['frameno'] == END_FRAME].set_index('marker_id')
    
    if start_data.empty or end_data.empty:
        raise ValueError("No data found for specified frames")
    
    common_markers = start_data.index.intersection(end_data.index)
    if common_markers.empty:
        raise ValueError("No common markers between frames")
        
    return (
        start_data.loc[common_markers, ['Xw', 'Yw', 'Zw']],
        end_data.loc[common_markers, ['Xw', 'Yw', 'Zw']]
    )

def create_3d_plot(start_pts, end_pts, output_path):
    """Create publication-quality 3D displacement plot"""
    vectors = end_pts.values - start_pts.values
    mean_disp = np.mean(np.linalg.norm(vectors, axis=1))
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot elements with enhanced styling
    ax.scatter(
        start_pts['Xw'], start_pts['Yw'], start_pts['Zw'],
        c='#1f77b4', marker='o', s=MARKER_SIZE, 
        edgecolors='k', linewidth=0.5, alpha=0.7,
        label=f'Frame {START_FRAME}'
    )
    
    ax.scatter(
        end_pts['Xw'], end_pts['Yw'], end_pts['Zw'],
        c='#d62728', marker='^', s=MARKER_SIZE+20,
        alpha=0.8, label=f'Frame {END_FRAME}'
    )
    
    ax.quiver(
        start_pts['Xw'], start_pts['Yw'], start_pts['Zw'],
        vectors[:,0], vectors[:,1], vectors[:,2],
        color='#2ca02c', arrow_length_ratio=0.1,
        linewidth=1.5, alpha=0.8, length=1.0
    )
    
    # Enhanced annotations
    ax.set_xlabel('X (mm)', fontsize=FONT_SIZE, labelpad=10)
    ax.set_ylabel('Y (mm)', fontsize=FONT_SIZE, labelpad=10)
    ax.set_zlabel('Z (mm)', fontsize=FONT_SIZE, labelpad=10)
    
    title = f'3D Marker Displacement\nMean Displacement: {mean_disp:.2f} mm'
    ax.set_title(title, fontsize=FONT_SIZE+2, pad=20)
    
    ax.legend(fontsize=FONT_SIZE-1, loc='upper right')
    ax.grid(False)
    set_axes_equal(ax)
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    return fig

def create_2d_plot(start_pts, end_pts, output_path):
    """Create publication-quality 2D displacement plot"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    vectors = end_pts.values - start_pts.values
    
    ax.scatter(
        start_pts['Xw'], start_pts['Yw'],
        c='#1f77b4', marker='o', s=MARKER_SIZE,
        edgecolors='k', linewidth=0.5, alpha=0.7,
        label=f'Frame {START_FRAME}'
    )
    
    ax.scatter(
        end_pts['Xw'], end_pts['Yw'],
        c='#d62728', marker='^', s=MARKER_SIZE+20,
        alpha=0.8, label=f'Frame {END_FRAME}'
    )
    
    ax.quiver(
        start_pts['Xw'], start_pts['Yw'],
        vectors[:,0], vectors[:,1],
        color='#2ca02c', angles='xy', scale_units='xy',
        scale=1, width=0.003, headwidth=3,
        headlength=5, headaxislength=4.5
    )
    
    ax.set_xlabel('X (mm)', fontsize=FONT_SIZE)
    ax.set_ylabel('Y (mm)', fontsize=FONT_SIZE)
    ax.set_title('2D Displacement (XY Plane)', fontsize=FONT_SIZE+2)
    
    ax.legend(fontsize=FONT_SIZE-1)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal')
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    return fig

# =============================================================================
# Main Program
# =============================================================================
def main():
    # Configure paths
    base_dir = os.path.join("Results", "data", "results")
    os.makedirs(base_dir, exist_ok=True)
    
    input_path = os.path.join(base_dir, 'marker_3d_coordinates.xlsx')
    output_3d = os.path.join(base_dir, '3D_displacement.png')
    output_2d = os.path.join(base_dir, '2D_displacement.png')
    
    try:
        # Load and process data
        start_pts, end_pts = load_and_process_data(input_path)
        print(f"Processed {len(start_pts)} common markers")
        
        # Generate plots
        print("Creating visualizations...")
        fig3d = create_3d_plot(start_pts, end_pts, output_3d)
        fig2d = create_2d_plot(start_pts, end_pts, output_2d)
        
        print(f"Saved 3D plot to: {output_3d}")
        print(f"Saved 2D plot to: {output_2d}")
        
        # Display plots
        plt.show()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting displacement visualization...")
    main()
    print("Visualization complete.")