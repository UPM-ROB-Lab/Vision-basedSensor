import cv2
import numpy as np
import pandas as pd 
import os
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy import ndimage
from scipy.signal import fftconvolve
import math
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class MarkerTracker:
    """A comprehensive marker tracking system for video analysis with distortion correction"""
    
    def __init__(self, config):
        """
        Initialize tracker with configuration parameters
        
        Args:
            config (dict): Configuration dictionary containing:
                - video_path: Path to input video
                - output_dir: Directory for saving results
                - crop_ratios: (left, right, top, bottom) crop ratios
                - marker_params: Parameters for marker detection
                - tracking_params: Parameters for marker tracking
        """
        self.config = config
        self._validate_config()
        self._setup_paths()
        self.frame_count = 0
        self.first_frame_markers = {}
        
    def _validate_config(self):
        """Validate configuration parameters"""
        required_keys = ['video_path', 'output_dir', 'crop_ratios']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
                
        if not os.path.exists(self.config['video_path']):
            raise FileNotFoundError(f"Video file not found: {self.config['video_path']}")
            
    def _setup_paths(self):
        """Set up output paths and directories"""
        os.makedirs(self.config['output_dir'], exist_ok=True)
        video_name = os.path.splitext(os.path.basename(self.config['video_path']))[0]
        self.output_csv = os.path.join(self.config['output_dir'], f'{video_name}_markers.csv')
        self.output_video = os.path.join(self.config['output_dir'], f'{video_name}_tracked.avi')
        
    def _init_video(self):
        """Initialize video capture and writer"""
        self.cap = cv2.VideoCapture(self.config['video_path'])
        if not self.cap.isOpened():
            raise IOError(f"Could not open video: {self.config['video_path']}")
            
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate crop dimensions
        left = int(self.width * self.config['crop_ratios'][0])
        right = self.width - int(self.width * self.config['crop_ratios'][1])
        top = int(self.height * self.config['crop_ratios'][2])
        bottom = self.height - int(self.height * self.config['crop_ratios'][3])
        self.crop_width = right - left
        self.crop_height = bottom - top
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(
            self.output_video, 
            fourcc, 
            self.fps, 
            (self.crop_width, self.crop_height)
        )
        
    def _preprocess_frame(self, frame):
        """Preprocess frame with cropping and optional undistortion"""
        # Crop frame
        left = int(self.width * self.config['crop_ratios'][0])
        right = self.width - int(self.width * self.config['crop_ratios'][1])
        top = int(self.height * self.config['crop_ratios'][2])
        bottom = self.height - int(self.height * self.config['crop_ratios'][3])
        cropped = frame[top:bottom, left:right]
        
        # Apply distortion correction if configured
        if 'calibration_params' in self.config:
            cropped = self._undistort_frame(cropped)
            
        return cropped
        
    def _undistort_frame(self, frame):
        """Apply camera calibration to undistort frame"""
        # Extract calibration parameters
        K = np.array(self.config['calibration_params']['camera_matrix'])
        D = np.array(self.config['calibration_params']['dist_coeffs'])
        
        # Get optimal new camera matrix
        h, w = frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            K, D, (w,h), 0, (w,h)
        )
        
        # Undistort
        map1, map2 = cv2.initUndistortRectifyMap(
            K, D, None, new_camera_matrix, (w,h), cv2.CV_16SC2
        )
        return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        
    @staticmethod
    def _find_markers(frame):
        """Detect circular markers in a frame using template matching"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Adaptive blurring based on image size
        if gray.shape[0] <= 480:  # Smaller resolution
            im_blur_3 = cv2.GaussianBlur(gray, (21, 21), 4.56)
            im_blur_8 = cv2.GaussianBlur(gray, (35, 35), 11.4)
            template = MarkerTracker._gkern(l=33, sig=7.4)
            thresh = 35
        else:  # Larger resolution
            im_blur_3 = cv2.GaussianBlur(gray, (39, 39), 8)
            im_blur_8 = cv2.GaussianBlur(gray, (101, 101), 20)
            template = MarkerTracker._gkern(l=80, sig=13)
            thresh = 20
            
        im_blur_sub = im_blur_8 - im_blur_3 + 15
        area_mask = cv2.inRange(im_blur_sub, thresh, 180 if gray.shape[0] <= 480 else 200)
        
        # Template matching for precise localization
        nrmcrimg = MarkerTracker._normxcorr2(template, area_mask) 
        mask = (nrmcrimg > 0.1).astype('uint8')
        
        return mask, area_mask
        
    @staticmethod
    def _gkern(l=5, sig=1.):
        """Create Gaussian kernel for template matching"""
        ax = np.linspace(-(l - 1)/2., (l - 1)/2., l)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5*(np.square(xx) + np.square(yy))/np.square(sig))
        return kernel/np.sum(kernel)
        
    @staticmethod
    def _normxcorr2(template, image, mode="same"):
        """Normalized cross-correlation for template matching"""
        if (np.ndim(template) > np.ndim(image) or 
            any(t > i for t, i in zip(template.shape, image.shape))):
            print("Warning: Template larger than image. Arguments may be swapped.")
            
        template = template - np.mean(template)
        image = image - np.mean(image)
        
        ar = np.flipud(np.fliplr(template))
        out = fftconvolve(image, ar.conj(), mode=mode)
        
        image_sq = fftconvolve(np.square(image), np.ones(template.shape), mode=mode)
        image_sq -= np.square(fftconvolve(image, np.ones(template.shape), mode=mode))/np.prod(template.shape)
        image_sq[image_sq < 0] = 0
        
        out = out/np.sqrt(image_sq * np.sum(np.square(template)))
        out[np.logical_not(np.isfinite(out))] = 0
        return out
        
    @staticmethod
    def _marker_center(mask, area_mask, frame=None):
        """Locate marker centers and fit ellipses with quality checks"""
        # Find local maxima as potential centers
        neighborhood_size = 8 if mask.shape[0] <= 480 else 14
        data_max = maximum_filter(mask, neighborhood_size)
        maxima = (mask == data_max)
        diff = ((data_max - minimum_filter(mask, neighborhood_size)) > 0)
        maxima[diff == 0] = 0
        
        labeled, num_objects = ndimage.label(maxima)
        if num_objects == 0:
            return []
            
        # Calculate centroids
        centers = np.array(ndimage.center_of_mass(mask, labeled, range(1, num_objects+1)))
        if centers.ndim == 1 and num_objects == 1:
            centers = centers.reshape(1, -1)
        if centers.size == 0:
            return []
            
        # Prepare contour detection
        if np.max(area_mask) > 1:
            area_mask_8u = area_mask.astype(np.uint8)
        else:
            area_mask_8u = (area_mask * 255).astype(np.uint8)
            
        # Clean contours with morphology
        kernel = np.ones((5,5), np.uint8)
        area_mask_8u = cv2.morphologyEx(area_mask_8u, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(area_mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        output = []
        centers_xy = [(c[1], c[0]) for c in centers] 
        unmatched = list(enumerate(centers_xy))
        
        # Match contours to centers with quality checks
        for contour in contours:
            if len(contour) < 5:
                continue
                
            # Fit ellipse
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (w, h), angle = ellipse
            
            # Determine major/minor axes
            if w > h:
                major, minor = w, h
                ellipse_angle = angle
            else:
                major, minor = h, w
                ellipse_angle = angle + 90
                
            if minor < 5:  # Minimum size threshold
                continue
                
            # Find best matching center within contour
            best_idx = -1
            min_dist = float('inf')
            threshold = (minor/10)**2  # Distance threshold squared
            
            for i, (orig_idx, (x, y)) in enumerate(unmatched):
                if cv2.pointPolygonTest(contour, (x,y), False) < 0:
                    continue
                    
                dist = (x-cx)**2 + (y-cy)**2
                if dist < threshold and dist < min_dist:
                    min_dist = dist
                    best_idx = i
                    
            if best_idx != -1:
                orig_idx, (x,y) = unmatched.pop(best_idx)
                output.append({
                    'center': (x,y),
                    'major_axis': float(major),
                    'minor_axis': float(minor),
                    'angle': float(ellipse_angle)
                })
                
                # Visualization if frame provided
                if frame is not None:
                    MarkerTracker._draw_marker(frame, (x,y), ellipse, major, minor, ellipse_angle)
                    
        return output
        
    @staticmethod
    def _draw_marker(frame, center, ellipse, major, minor, angle):
        """Visualize marker with ellipse and axes"""
        cv2.ellipse(frame, ellipse, (0,255,0), 2)
        
        angle_rad = math.radians(angle)
        cx, cy = center
        
        # Draw major axis (yellow)
        maj_p1 = (cx - major/2 * math.cos(angle_rad), 
                 cy - major/2 * math.sin(angle_rad))
        maj_p2 = (cx + major/2 * math.cos(angle_rad), 
                 cy + major/2 * math.sin(angle_rad))
        cv2.line(frame, (int(maj_p1[0]), int(maj_p1[1])),
                (int(maj_p2[0]), int(maj_p2[1])), (0,255,255), 2)
                
        # Draw minor axis (blue)
        min_p1 = (cx - minor/2 * math.cos(angle_rad + math.pi/2),
                 cy - minor/2 * math.sin(angle_rad + math.pi/2))
        min_p2 = (cx + minor/2 * math.cos(angle_rad + math.pi/2),
                 cy + minor/2 * math.sin(angle_rad + math.pi/2))
        cv2.line(frame, (int(min_p1[0]), int(min_p1[1])),
                (int(min_p2[0]), int(min_p2[1])), (255,0,0), 2)
                
    def _process_first_frame(self, markers):
        """Establish marker identities in first frame"""
        if not markers:
            raise ValueError("No markers detected in first frame!")
            
        centers = np.array([m['center'] for m in markers])
        
        # 1. Find center marker
        center_pos = np.mean(centers, axis=0)
        dists = np.linalg.norm(centers - center_pos, axis=1)
        center_idx = np.argmin(dists)
        center_marker = markers[center_idx]
        
        # Store center (layer 0, angle 0)
        self.first_frame_markers[(0,0)] = {
            **center_marker,
            'Ox': center_marker['center'][0],
            'Oy': center_marker['center'][1]
        }
        
        # 2. Process remaining markers in concentric rings
        remaining = [m for i,m in enumerate(markers) if i != center_idx]
        if not remaining:
            return
            
        remaining_centers = np.array([m['center'] for m in remaining])
        vectors = remaining_centers - center_marker['center']
        
        # Polar coordinates
        distances = np.linalg.norm(vectors, axis=1)
        angles = np.arctan2(vectors[:,1], vectors[:,0])  # [-π, π]
        
        # Cluster into layers by distance
        kmeans = KMeans(n_clusters=self.config.get('num_layers', 5), n_init=10)
        kmeans.fit(distances.reshape(-1,1))
        
        # Sort layers by radius
        layer_order = np.argsort(kmeans.cluster_centers_.flatten())
        layer_map = {orig: new+1 for new, orig in enumerate(layer_order)}  # Layer 0 is center
        
        # Assign markers to layers and angles
        for i, m in enumerate(remaining):
            layer = layer_map[kmeans.labels_[i]]
            angle_idx = -1  # Temporary placeholder
            
            # Store with polar info
            self.first_frame_markers[(layer, angle_idx)] = {
                **m,
                'angle_rad': angles[i],
                'Ox': m['center'][0],
                'Oy': m['center'][1]
            }
            
        # Sort markers in each layer by angle and assign angle indices
        for layer in range(1, self.config.get('num_layers',5)+1):
            layer_markers = [(k,v) for k,v in self.first_frame_markers.items() 
                           if k[0] == layer and k[1] == -1]
                           
            if not layer_markers:
                continue
                
            # Sort by angle (CCW from positive x-axis)
            layer_markers.sort(key=lambda x: x[1]['angle_rad'])
            
            # Find index of marker closest to 0 angle
            start_idx = np.argmin([abs(m['angle_rad']) for _,m in layer_markers])
            
            # Assign angle indices (0 at start_idx, increasing CCW)
            for i, ((_,_), m) in enumerate(layer_markers):
                angle_idx = (i - start_idx) % len(layer_markers)
                # Update with correct angle index
                del self.first_frame_markers[(layer, -1)]
                self.first_frame_markers[(layer, angle_idx)] = m
                
    def _track_markers(self, frame, markers):
        """Track markers relative to first frame positions"""
        if not self.first_frame_markers or not markers:
            return []
            
        # Current marker positions
        current_positions = {tuple(m['center']): m for m in markers}
        current_centers = np.array([m['center'] for m in markers])
        
        tracked_data = []
        min_dist = self.config.get('min_marker_distance', 20)
        
        # Match each reference marker to nearest current marker
        for (layer, angle), ref in self.first_frame_markers.items():
            ref_pos = np.array([ref['Ox'], ref['Oy']])
            
            if len(current_centers) == 0:
                continue
                
            # Find closest current marker
            dists = cdist([ref_pos], current_centers)[0]
            closest_idx = np.argmin(dists)
            
            if dists[closest_idx] > min_dist:
                continue
                
            closest_pos = tuple(current_centers[closest_idx])
            curr = current_positions.get(closest_pos)
            
            if curr:
                # Record tracking data
                tracked_data.append({
                    'frameno': self.frame_count,
                    'row': layer,
                    'col': angle,
                    'Ox': ref['Ox'],
                    'Oy': ref['Oy'],
                    'Cx': curr['center'][0],
                    'Cy': curr['center'][1],
                    'major_axis': curr['major_axis'],
                    'minor_axis': curr['minor_axis'],
                    'angle': curr['angle']
                })
                
                # Draw tracking visualization
                self._draw_tracking(frame, ref, curr)
                
        return tracked_data
        
    def _draw_tracking(self, frame, ref, curr):
        """Draw marker tracking visualization"""
        cx, cy = curr['center']
        ox, oy = ref['Ox'], ref['Oy']
        
        # Marker center
        cv2.circle(frame, (int(cx), int(cy)), 4, (0,0,255), -1)
        
        # Displacement vector
        cv2.arrowedLine(frame, (int(ox), int(oy)), (int(cx), int(cy)),
                       (0,0,255), 2, tipLength=0.25)
        
        # Major/minor axes
        angle_rad = np.deg2rad(curr['angle'])
        maj_len = curr['major_axis'] / 2
        min_len = curr['minor_axis'] / 2
        
        # Major axis (yellow)
        maj_p1 = (int(cx - maj_len * np.cos(angle_rad)),
                 int(cy - maj_len * np.sin(angle_rad)))
        maj_p2 = (int(cx + maj_len * np.cos(angle_rad)),
                 int(cy + maj_len * np.sin(angle_rad)))
        cv2.line(frame, maj_p1, maj_p2, (0,255,255), 2)
        
        # Minor axis (blue)
        min_p1 = (int(cx - min_len * np.cos(angle_rad + np.pi/2)),
                 int(cy - min_len * np.sin(angle_rad + np.pi/2)))
        min_p2 = (int(cx + min_len * np.cos(angle_rad + np.pi/2)),
                 int(cy + min_len * np.sin(angle_rad + np.pi/2)))
        cv2.line(frame, min_p1, min_p2, (255,0,0), 2)
        
    def process(self):
        """Main processing loop for video tracking"""
        self._init_video()
        data = []
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Preprocess and detect markers
            cropped = self._preprocess_frame(frame)
            mask, area_mask = self._find_markers(cropped)
            markers = self._marker_center(mask, area_mask, cropped.copy())
            
            # First frame processing
            if self.frame_count == 0:
                self._process_first_frame(markers)
                
            # Track markers and collect data
            frame_data = self._track_markers(cropped, markers)
            data.extend(frame_data)
            
            # Write frame to output video
            self.writer.write(cropped)
            self.frame_count += 1
            
            # Progress feedback
            if self.frame_count % 100 == 0:
                print(f"Processed frame {self.frame_count}")
                
        # Save results
        self._save_results(data)
        self._cleanup()
        
    def _save_results(self, data):
        """Save tracking data to CSV"""
        df = pd.DataFrame(data)
        df.to_csv(self.output_csv, index=False)
        print(f"Saved tracking data to {self.output_csv}")
        
    def _cleanup(self):
        """Release resources"""
        self.cap.release()
        self.writer.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    config = {
        'video_path': './video/test2.avi',
        'output_dir': './results',
        'crop_ratios': (1/8, 1/8, 1/16, 0),  # left, right, top, bottom
        'num_layers': 5,  # Number of concentric marker circles
        'min_marker_distance': 20,  # Minimum distance between markers (pixels)
        # Optional camera calibration parameters:
        # 'calibration_params': {
        #     'camera_matrix': [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        #     'dist_coeffs': [k1, k2, p1, p2, k3]
        # }
    }
    
    tracker = MarkerTracker(config)
    tracker.process()