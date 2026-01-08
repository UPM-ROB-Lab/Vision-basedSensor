import cv2
import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from marker_circle import find_marker, marker_center 
import math

def process_video():
    """Automatically process video to track circular marker array and save trajectory data.
    
    Output CSV columns:
    - frameno: Frame number
    - row: Layer index (0=center, increasing outward)
    - col: Angular position index (0 starts at positive x-axis, increments CCW)
    - Ox,Oy: Original position in first frame (reference)
    - Cx,Cy: Current position
    - major_axis,minor_axis: Ellipse parameters
    - angle: Ellipse orientation angle
    
    Visual output shows:
    - Detected markers (red)
    - Displacement vectors (red arrows)  
    - Major/minor axes (yellow/blue)
    """
    
    # --------- Configuration ---------
    VIDEO_DIR = './video'
    INPUT_VIDEO = os.path.join(VIDEO_DIR, 'test2.avi')
    OUTPUT_VIDEO = os.path.join(VIDEO_DIR, 'exp_processed.avi') 
    OUTPUT_CSV = os.path.join(VIDEO_DIR, 'marker_locations_0.csv')
    
    # Crop ratios - adjust based on marker array geometry
    CROP_RATIOS = (1/8, 1/8, 1/16, 0)  # Left, Right, Top, Bottom
    
    # Marker tracking parameters
    MIN_MARKER_DISTANCE = 20  # Minimum distance between markers (pixels)
    NUM_OUTER_LAYERS = 5      # Number of concentric circles in array
    
    # Visualization settings
    MARKER_COLOR = (0, 0, 255)    # Red
    VECTOR_COLOR = (0, 0, 255)    # Red 
    MAJOR_AXIS_COLOR = (0, 255, 0) # Green
    MINOR_AXIS_COLOR = (255, 0, 0) # Blue
    
    # --------- Setup ---------
    os.makedirs(VIDEO_DIR, exist_ok=True)
    
    # Initialize video capture
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {INPUT_VIDEO}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate crop dimensions
    left = int(width * CROP_RATIOS[0])
    right = width - int(width * CROP_RATIOS[1]) 
    top = int(height * CROP_RATIOS[2])
    bottom = height - int(height * CROP_RATIOS[3])
    crop_width = right - left
    crop_height = bottom - top
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (crop_width, crop_height))
    
    # Data storage
    data = {
        'frameno': [],
        'row': [],   # Layer index (0=center)
        'col': [],   # Angular index
        'Ox': [],    # Original X
        'Oy': [],    # Original Y  
        'Cx': [],    # Current X
        'Cy': [],    # Current Y
        'major_axis': [],
        'minor_axis': [],
        'angle': []
    }
    
    first_frame_markers = {}  # Stores marker positions from first frame
    
    # --------- Processing Loop ---------
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop frame
        cropped = frame[top:bottom, left:right]
        
        # Detect markers
        mask, area_mask = find_marker(cropped)
        markers = marker_center(mask, area_mask, cropped)
        
        # Initialize visualization
        vis_frame = cropped.copy()
        
        # First frame processing - establish marker identities
        if frame_count == 0:
            if not markers:
                raise ValueError("No markers detected in first frame!")
            
            # Convert to numpy array for calculations
            centers = np.array([m['center'] for m in markers])
            
            # Step 1: Identify center marker
            mean_center = np.mean(centers, axis=0)
            dist_to_mean = np.linalg.norm(centers - mean_center, axis=1)
            center_idx = np.argmin(dist_to_mean)
            center_pos = centers[center_idx]
            
            # Store center marker (layer 0, angle 0)
            first_frame_markers[(0, 0)] = {
                **markers[center_idx],
                'Ox': center_pos[0],
                'Oy': center_pos[1]
            }
            
            # Process remaining markers
            remaining = [m for i, m in enumerate(markers) if i != center_idx]
            if remaining:
                # Calculate polar coordinates relative to center
                remaining_centers = np.array([m['center'] for m in remaining])
                vectors = remaining_centers - center_pos
                distances = np.linalg.norm(vectors, axis=1)
                angles = np.arctan2(vectors[:,1], vectors[:,0])  # [-π, π]
                
                # Cluster markers into layers by distance
                kmeans = KMeans(n_clusters=NUM_OUTER_LAYERS, n_init=10)
                kmeans.fit(distances.reshape(-1, 1))
                
                # Sort layers by radius
                layer_order = np.argsort(kmeans.cluster_centers_.flatten())
                layer_map = {orig: new for new, orig in enumerate(layer_order)}
                
                # Assign markers to layers
                for i, m in enumerate(remaining):
                    # Get original cluster assignment
                    orig_layer = kmeans.labels_[i]
                    # Get sorted layer index (1-based, since 0 is center)
                    layer_idx = layer_map[orig_layer] + 1
                    
                    # Store angle with marker data
                    m['angle_rad'] = angles[i]
                    m['Ox'] = m['center'][0]
                    m['Oy'] = m['center'][1]
                    
                    # Temporarily store with placeholder angle_idx
                    first_frame_markers[(layer_idx, -1)] = m
                
                # For each layer, sort markers by angle and assign angle indices
                for layer in range(1, NUM_OUTER_LAYERS + 1):
                    # Get all markers in this layer
                    layer_markers = [(k, v) for k, v in first_frame_markers.items() 
                                   if k[0] == layer and k[1] == -1]
                    
                    if not layer_markers:
                        continue
                        
                    # Sort by angle (CCW from positive x-axis)
                    layer_markers.sort(key=lambda x: x[1]['angle_rad'])
                    
                    # Find index of marker closest to 0 angle
                    start_idx = np.argmin([abs(m['angle_rad']) for _, m in layer_markers])
                    
                    # Assign angle indices (0 is at start_idx, increasing CCW)
                    for i, ((_, _), m) in enumerate(layer_markers):
                        angle_idx = (i - start_idx) % len(layer_markers)
                        # Remove temp entry and create new with correct angle_idx
                        del first_frame_markers[(layer, -1)]
                        first_frame_markers[(layer, angle_idx)] = m
        
        # For all frames (including first), track markers
        if first_frame_markers and markers:
            # Create lookup of current marker positions
            current_positions = {tuple(m['center']): m for m in markers}
            current_centers = np.array([m['center'] for m in markers])
            
            # Match each reference marker to nearest current marker
            for (layer, angle), ref_marker in first_frame_markers.items():
                ref_pos = np.array([ref_marker['Ox'], ref_marker['Oy']])
                
                if len(current_centers) == 0:
                    continue
                
                # Find closest current marker
                distances = cdist([ref_pos], current_centers)[0]
                closest_idx = np.argmin(distances)
                closest_pos = tuple(current_centers[closest_idx])
                
                # Verify minimum distance
                if distances[closest_idx] > MIN_MARKER_DISTANCE:
                    continue
                
                current_marker = current_positions.get(closest_pos)
                if current_marker:
                    # Record data
                    data['frameno'].append(frame_count)
                    data['row'].append(layer)
                    data['col'].append(angle)
                    data['Ox'].append(ref_marker['Ox'])
                    data['Oy'].append(ref_marker['Oy'])
                    data['Cx'].append(current_marker['center'][0])
                    data['Cy'].append(current_marker['center'][1])
                    data['major_axis'].append(current_marker['major_axis'])
                    data['minor_axis'].append(current_marker['minor_axis'])
                    data['angle'].append(current_marker['angle'])
                    
                    # Draw visualization
                    cx, cy = current_marker['center']
                    ox, oy = ref_marker['Ox'], ref_marker['Oy']
                    
                    # Marker center
                    cv2.circle(vis_frame, (int(cx), int(cy)), 4, MARKER_COLOR, -1)
                    
                    # Displacement vector
                    cv2.arrowedLine(vis_frame, (int(ox), int(oy)), (int(cx), int(cy)),
                                   VECTOR_COLOR, 2, tipLength=0.25)
                    
                    # Draw major/minor axes
                    angle_rad = np.deg2rad(current_marker['angle'])
                    maj_len = current_marker['major_axis'] / 2
                    min_len = current_marker['minor_axis'] / 2
                    
                    # Major axis (yellow)
                    maj_p1 = (int(cx - maj_len * np.cos(angle_rad)),
                             int(cy - maj_len * np.sin(angle_rad)))
                    maj_p2 = (int(cx + maj_len * np.cos(angle_rad)),
                             int(cy + maj_len * np.sin(angle_rad)))
                    cv2.line(vis_frame, maj_p1, maj_p2, (0, 255, 255), 2)
                    
                    # Minor axis (blue)
                    min_p1 = (int(cx - min_len * np.cos(angle_rad + np.pi/2)),
                             int(cy - min_len * np.sin(angle_rad + np.pi/2)))
                    min_p2 = (int(cx + min_len * np.cos(angle_rad + np.pi/2)),
                             int(cy + min_len * np.sin(angle_rad + np.pi/2)))
                    cv2.line(vis_frame, min_p1, min_p2, (255, 0, 0), 2)
        
        # Write frame
        out.write(vis_frame)
        frame_count += 1
    
    # --------- Cleanup ---------
    cap.release()
    out.release()
    
    # Save data
    pd.DataFrame(data).to_csv(OUTPUT_CSV, index=False)
    print(f"Processing complete. Results saved to {VIDEO_DIR}")

if __name__ == '__main__':
    process_video()