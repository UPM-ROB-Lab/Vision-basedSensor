import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Global plot settings (SCI style)
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
    'savefig.dpi': 300
})

def crop_image(img, ratios=(1/8, 1/8, 1/16, 0)):
    """Crop image with given ratios (left, right, top, bottom)"""
    h, w = img.shape[:2]
    left = int(w * ratios[0])
    right = int(w * ratios[1])
    top = int(h * ratios[2])
    bottom = int(h * ratios[3])
    return img[top:h-bottom, left:w-right]

def save_calib_results(mtx, dist, error, path):
    """Save calibration results to Excel"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    data = [
        ['fx', mtx[0,0], 'Focal length x'],
        ['fy', mtx[1,1], 'Focal length y'],
        ['cx', mtx[0,2], 'Principal point x'],
        ['cy', mtx[1,2], 'Principal point y'],
        ['skew', mtx[0,1], 'Skew coefficient'],
        ['k1', dist[0], 'Radial dist coeff 1'],
        ['k2', dist[1], 'Radial dist coeff 2'],
        ['p1', dist[2], 'Tangential dist coeff 1'],
        ['p2', dist[3], 'Tangential dist coeff 2'],
        ['k3', dist[4], 'Radial dist coeff 3'],
        ['Reproj Error', error, 'Mean error (px)']
    ]
    
    pd.DataFrame(data, columns=['Param','Value','Desc']).to_excel(path, index=False)

def calibrate_camera(img_dir, pattern_size, square_size, show_corners=False):
    """Perform camera calibration"""
    print(f"Processing images in: {img_dir}")
    
    # Prepare 3D points
    objp = np.zeros((np.prod(pattern_size), 3), np.float32)
    objp[:,:2] = np.mgrid[:pattern_size[0], :pattern_size[1]].T.reshape(-1,2) * square_size
    
    obj_points = []
    img_points = []
    valid_imgs = []
    img_size = None
    
    for f in [f for f in os.listdir(img_dir) if f.lower().endswith(('.png','.jpg'))]:
        img = cv2.imread(os.path.join(img_dir, f))
        if img is None: continue
        
        # Crop and get size
        img_crop = crop_image(img)
        if img_size is None: img_size = img_crop.shape[1::-1]
            
        # Find corners
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size)
        
        if ret:
            obj_points.append(objp)
            corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), 
                                     (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            img_points.append(corners)
            valid_imgs.append(f)
            
            if show_corners:
                cv2.drawChessboardCorners(img_crop, pattern_size, corners, ret)
                cv2.imshow('Corners', cv2.resize(img_crop, (0,0), fx=0.5, fy=0.5))
                cv2.waitKey(100)
    
    if show_corners: cv2.destroyAllWindows()
    
    if len(obj_points) < 3:
        print("Insufficient valid images")
        return None
    
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None)
    
    return {
        'mtx': mtx,
        'dist': dist.flatten(),
        'error': ret,
        'obj_points': obj_points,
        'img_points': img_points,
        'rvecs': rvecs,
        'tvecs': tvecs,
        'valid_imgs': valid_imgs
    }

def plot_comparison(img_path, mtx, dist, error):
    """Plot original vs undistorted image"""
    img = cv2.imread(img_path)
    if img is None: return
    
    img_crop = crop_image(img)
    h, w = img_crop.shape[:2]
    
    # Undistort
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    undist = cv2.undistort(img_crop, mtx, dist, None, new_mtx)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    
    for ax, im, title in zip([ax1, ax2], 
                            [img_crop, undist], 
                            ['(a) Original', '(b) Undistorted']):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontweight='bold')
        ax.axis('off')
        for y in range(h//10, h, h//10):
            ax.axhline(y, color='r' if ax==ax1 else 'g', ls='--', lw=1, alpha=0.6)
    
    fig.suptitle("Calibration Results", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_3d_poses(rvecs, tvecs, pattern_size, square_size):
    """3D visualization of camera poses"""
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Camera visualization
    scale = square_size * 2
    verts = np.array([[0,0,0], [-scale,-scale,scale*1.5], 
                      [scale,-scale,scale*1.5], [scale,scale,scale*1.5], 
                      [-scale,scale,scale*1.5]])
    faces = [[0,1,2], [0,2,3], [0,3,4], [0,4,1], [1,2,3,4]]
    ax.add_collection3d(Poly3DCollection(
        [verts[f] for f in faces], 
        facecolors='crimson', edgecolors='k', alpha=0.4, linewidths=0.8))
    
    # Board visualization
    objp = np.zeros((np.prod(pattern_size), 3), np.float32)
    objp[:,:2] = np.mgrid[:pattern_size[0], :pattern_size[1]].T.reshape(-1,2) * square_size
    
    all_points = []
    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        R = cv2.Rodrigues(rvec)[0]
        pts = (R @ objp.T + tvec).T
        all_points.append(pts)
        
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c='steelblue', s=2, alpha=0.6)
        corners = pts[[0, pattern_size[0]-1, -1, -pattern_size[0], 0]]
        ax.plot(corners[:,0], corners[:,1], corners[:,2], c='navy', lw=0.8, alpha=0.7)
        
        center = pts.mean(axis=0)
        ax.text(center[0], center[1], center[2], str(i+1), fontsize=9, fontweight='bold')
    
    # Formatting
    all_points.append(verts)
    X,Y,Z = np.vstack(all_points).T
    max_range = max(X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()) / 2
    
    ax.set(
        xlim=(X.mean()-max_range, X.mean()+max_range),
        ylim=(Y.mean()-max_range, Y.mean()+max_range),
        zlim=(Z.mean()-max_range, Z.mean()+max_range),
        xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)',
        title='3D Camera Poses Visualization'
    )
    ax.view_init(elev=-60, azim=-90)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Configuration
    IMG_DIR = 'calibration_images'
    PATTERN = (6, 6)
    SQUARE_SIZE = 3.0
    OUTPUT_FILE = os.path.join('Results', 'data', 'calibration_results.xlsx')
    
    # Run calibration
    results = calibrate_camera(IMG_DIR, PATTERN, SQUARE_SIZE)
    
    if results:
        print(f"\nCalibration successful (Error: {results['error']:.4f} px)")
        
        # Save results
        save_calib_results(results['mtx'], results['dist'], 
                          results['error'], OUTPUT_FILE)
        
        # Generate plots
        plot_comparison(
            os.path.join(IMG_DIR, results['valid_imgs'][0]),
            results['mtx'],
            results['dist'],
            results['error']
        )
        
        plot_3d_poses(
            results['rvecs'],
            results['tvecs'],
            PATTERN,
            SQUARE_SIZE
        )
    else:
        print("Calibration failed")