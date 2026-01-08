import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse

def process_marker_data(txt_path, output_path):
    """
    Process raw marker data from TXT file and save averaged results to CSV
    Args:
        txt_path: Input TXT file path
        output_path: Output CSV path
    Returns:
        bool: True if successful
    """
    print(f"Processing data from: {txt_path}")
    
    try:
        # Load and parse data
        cols = [0, 1, 2, 5, 6, 7, 8, 9]
        col_names = ['frameno', 'row', 'col', 'Cx', 'Cy', 'major', 'minor', 'angle']
        df = pd.read_csv(txt_path, header=None, skiprows=1, sep='\s+',
                        usecols=cols, names=col_names, engine='python')
        
        # Clean data
        for col in col_names:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip(',')
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        
        # Calculate averages
        avg_shape = df.groupby(['row', 'col'])[['major', 'minor', 'angle']].mean().reset_index()
        avg_pos = df.groupby(['row', 'col'])[['Cx', 'Cy']].mean().reset_index()
        
        # Merge and format results
        result = pd.merge(avg_shape, avg_pos, on=['row', 'col'])
        result = result.sort_values(['row', 'col'])
        result['id'] = range(1, len(result)+1)
        
        # Save output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result[['id', 'Cx', 'Cy', 'major', 'minor', 'angle']].rename(columns={
            'Cx': 'u', 'Cy': 'v'
        }).to_csv(output_path, index=False, float_format='%.4f')
        
        print(f"Saved processed data to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return False

def plot_markers(csv_path):
    """
    Plot marker positions and shapes from processed CSV
    Args:
        csv_path: Path to processed CSV file
    """
    print(f"Plotting data from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Scatter plot with color mapping
        sc = ax.scatter(df['u'], df['v'], c=df['minor'], s=10, 
                       cmap='plasma', alpha=0.8, edgecolors='k', linewidth=0.5)
        plt.colorbar(sc).set_label('Minor Axis (px)')
        
        # Draw ellipses and major axes
        for _, row in df.iterrows():
            angle_rad = np.deg2rad(row['angle'])
            dx = (row['major']/2) * np.cos(angle_rad)
            dy = (row['major']/2) * np.sin(angle_rad)
            
            ax.add_patch(Ellipse(
                (row['u'], row['v']), row['major'], row['minor'], row['angle'],
                facecolor='none', edgecolor='green', linewidth=1.5, alpha=0.7))
            
            ax.plot([row['u']-dx, row['u']+dx], [row['v']-dy, row['v']+dy],
                   color='red', linewidth=1)
            
            ax.text(row['u']+row['major']*0.6, row['v'], 
                   str(row['id']), fontsize=8)
        
        # Format plot
        ax.set(xlabel='u (px)', ylabel='v (px)', 
               title='Marker Positions and Shapes', aspect='equal')
        ax.invert_yaxis()
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error plotting data: {str(e)}")

if __name__ == "__main__":
    data_dir = os.path.join("Results", "data", "PreprocessPara")
    os.makedirs(data_dir, exist_ok=True)
    
    in_file = os.path.join(data_dir, "MarkerCalibration.txt")
    out_file = os.path.join(data_dir, "pixel_marker.csv")
    
    if process_marker_data(in_file, out_file):
        plot_markers(out_file)
    else:
        print("Processing failed - skipping plot")