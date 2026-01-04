import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- 全局绘图设置 (SCI 风格 + 加粗) ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'          # 全局字体加粗
plt.rcParams['axes.labelweight'] = 'bold'     # 坐标轴标签加粗
plt.rcParams['axes.titleweight'] = 'bold'     # 标题加粗
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['savefig.dpi'] = 300

def cresize_crop_mini(img, crop_ratios=(1/8, 1/8, 1/16, 0)): 
    """
    环形阵列裁剪系数 (左, 右, 上, 下)
    """
    h_original, w_original = img.shape[:2]
    left = int(w_original * crop_ratios[0])
    right = int(w_original * crop_ratios[1])
    top = int(h_original * crop_ratios[2])
    bottom = int(h_original * crop_ratios[3])
    
    start_row, end_row = top, h_original - bottom
    start_col, end_col = left, w_original - right
    
    cropped_img = img[start_row:end_row, start_col:end_col]
    new_h, new_w = cropped_img.shape[:2]
    return cropped_img, new_w, new_h

def save_results_to_excel(camera_matrix, dist_coeffs, reproj_error, filepath):
    output_dir = os.path.dirname(filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    data_rows = [
        ['fx', camera_matrix[0, 0], 'Focal length x'],
        ['fy', camera_matrix[1, 1], 'Focal length y'],
        ['cx', camera_matrix[0, 2], 'Principal point x'],
        ['cy', camera_matrix[1, 2], 'Principal point y'],
        ['skew', camera_matrix[0, 1], 'Skew coefficient'],
        ['---', '---', '---']
    ]
    
    dist_labels = ['k1', 'k2', 'p1', 'p2', 'k3']
    for i, coeff in enumerate(dist_coeffs.flatten()):
        label = dist_labels[i] if i < len(dist_labels) else f'dist_{i}'
        data_rows.append([label, coeff, 'Distortion coeff'])
        
    data_rows.append(['---', '---', '---'])
    data_rows.append(['Reprojection Error', reproj_error, 'Mean error (pixels)'])
    
    df = pd.DataFrame(data_rows, columns=['Parameter', 'Value', 'Description'])
    try:
        df.to_excel(filepath, index=False, engine='openpyxl')
        print(f"Results saved to: {filepath}")
    except Exception as e:
        print(f"Error saving Excel: {e}")

def calibrate_from_images(image_folder, pattern_size, square_size_mm, show_corners=False):
    print(f"--- Processing images in '{image_folder}' ---")
    
    # 准备 3D 坐标
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp = objp * square_size_mm

    obj_points = [] 
    img_points = [] 
    
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print("No images found.")
        return None
    
    image_size = None
    valid_count = 0

    for fname in image_files:
        img_original = cv2.imread(fname)
        if img_original is None: continue
        
        # 裁剪
        img, new_w, new_h = cresize_crop_mini(img_original)
        if image_size is None: image_size = (new_w, new_h)
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            valid_count += 1
            obj_points.append(objp)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners2)
            
            if show_corners:
                cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
                cv2.imshow('Corners', cv2.resize(img, (new_w//2, new_h//2)))
                cv2.waitKey(100)
        else:
            print(f"Corners not found in {os.path.basename(fname)}")

    if show_corners: cv2.destroyAllWindows()
    
    if valid_count < 3:
        print("Not enough valid images for calibration.")
        return None

    print(f"Calibrating with {valid_count} images...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)
    
    # 返回所有需要的数据用于绘图
    return mtx, dist, ret, obj_points, img_points, rvecs, tvecs

# --- 绘图函数 1: 2D 对比图 ---
def plot_calibration_comparison(img_path, mtx, dist, error):
    """绘制 SCI 风格的 原图 vs 去畸变图 对比"""
    img = cv2.imread(img_path)
    if img is None: return
    
    # 裁剪
    img_cropped, w, h = cresize_crop_mini(img)
    
    # 去畸变
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img_cropped, mtx, dist, None, new_camera_matrix)
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图
    axes[0].imshow(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
    axes[0].set_title('(a) Original Image', fontweight='bold')
    axes[0].axis('off')
    # 画参考线
    for y in range(h//10, h, h//10):
        axes[0].axhline(y, color='red', linestyle='--', linewidth=1.0, alpha=0.6)

    # 右图
    axes[1].imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    axes[1].set_title('(b) Undistorted Image', fontweight='bold')
    axes[1].axis('off')
    # 画参考线
    for y in range(h//10, h, h//10):
        axes[1].axhline(y, color='green', linestyle='--', linewidth=1.0, alpha=0.6)

    # 误差值显示在右图右下角
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    text_str = f"Mean Reprojection Error:\n{error:.4f} pixels"
    axes[1].text(0.97, 0.03, text_str, transform=axes[1].transAxes, fontsize=11,
                 verticalalignment='bottom', horizontalalignment='right', bbox=props, fontweight='bold')

    plt.suptitle("Calibration Result Comparison", fontsize=14, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.show()

# --- 绘图函数 2: 3D 位姿图 (修改版：无网格，无误差文字) ---
def plot_3d_poses(rvecs, tvecs, pattern_size, square_size, error):
    """绘制 SCI 风格的 3D 相机与标定板位置关系图"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 【修改】移除网格线和背景板，实现干净的白色背景
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    # 1. 绘制相机 (红色金字塔)
    cam_scale = square_size * 2
    # 顶点: 镜头中心(0,0,0) 和 底面四个点
    vs = np.array([[0,0,0], 
                   [-cam_scale, -cam_scale, cam_scale*1.5],
                   [cam_scale, -cam_scale, cam_scale*1.5],
                   [cam_scale, cam_scale, cam_scale*1.5],
                   [-cam_scale, cam_scale, cam_scale*1.5]])
    faces = [[vs[0],vs[1],vs[2]], [vs[0],vs[2],vs[3]], 
             [vs[0],vs[3],vs[4]], [vs[0],vs[4],vs[1]], 
             [vs[1],vs[2],vs[3],vs[4]]]
    
    ax.add_collection3d(Poly3DCollection(faces, facecolors='crimson', linewidths=0.8, edgecolors='k', alpha=0.4))
    ax.text(0, 0, -cam_scale*0.8, "Camera", color='darkred', ha='center', fontsize=11, fontweight='bold')

    # 2. 绘制标定板
    w, h = pattern_size
    # 标定板局部坐标
    objp = np.zeros((h * w, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    objp = objp * square_size

    all_points = [] # 用于计算坐标轴范围

    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        R, _ = cv2.Rodrigues(rvec)
        # 坐标变换: P_cam = R * P_board + T
        board_points = (np.dot(R, objp.T) + tvec).T
        all_points.append(board_points)
        
        # 绘制点
        ax.scatter(board_points[:,0], board_points[:,1], board_points[:,2], c='steelblue', s=2, alpha=0.6)
        
        # 绘制边框
        corners = board_points[[0, w-1, w*h-1, w*(h-1), 0], :]
        ax.plot(corners[:,0], corners[:,1], corners[:,2], color='navy', linewidth=0.8, alpha=0.7)
        
        # 标记序号
        center = np.mean(board_points, axis=0)
        ax.text(center[0], center[1], center[2], str(i+1), fontsize=9, color='black', fontweight='bold')

    # 3. 设置坐标轴比例 (Equal Aspect Ratio)
    all_points = np.vstack(all_points)
    all_points = np.vstack([all_points, vs]) 
    
    X, Y, Z = all_points[:,0], all_points[:,1], all_points[:,2]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x, mid_y, mid_z = (X.max()+X.min())*0.5, (Y.max()+Y.min())*0.5, (Z.max()+Z.min())*0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 4. 标签与美化
    ax.set_xlabel('X (mm)', labelpad=10, fontweight='bold')
    ax.set_ylabel('Y (mm)', labelpad=10, fontweight='bold')
    ax.set_zlabel('Z (mm)', labelpad=10, fontweight='bold')
    ax.set_title("3D Extrinsic Parameters Visualization", fontsize=14, pad=20, fontweight='bold')
    
    # 调整视角 (俯视)
    ax.view_init(elev=-60, azim=-90)
    plt.tight_layout()
    plt.show()

# --- 主程序 ---
if __name__ == "__main__":
    # 配置参数
    IMAGE_FOLDER = 'calibration_images'
    PATTERN_SIZE = (6, 6)  # 内部角点数
    SQUARE_SIZE_MM = 3.0   # 棋盘格边长
    OUTPUT_FILE = os.path.join('Results', 'data', 'IntrinsicParameters.xlsx')
    
    # 1. 检查文件夹
    if not os.path.isdir(IMAGE_FOLDER):
        print(f"Error: Folder '{IMAGE_FOLDER}' not found.")
        exit()

    # 2. 执行标定
    result = calibrate_from_images(IMAGE_FOLDER, PATTERN_SIZE, SQUARE_SIZE_MM)
    
    if result:
        mtx, dist, err, obj_pts, img_pts, rvecs, tvecs = result
        
        print("\n--- Calibration Successful ---")
        print(f"Mean Error: {err:.4f} px")
        
        # 3. 保存数据
        save_results_to_excel(mtx, dist, err, OUTPUT_FILE)
        
        # 4. 绘制图表 (SCI 风格)
        print("\nGenerating Figures...")
        
        # 图1: 2D 对比 (使用第一张图作为示例)
        sample_img = os.path.join(IMAGE_FOLDER, os.listdir(IMAGE_FOLDER)[0])
        plot_calibration_comparison(sample_img, mtx, dist, err)
        
        # 图2: 3D 位姿 (无网格，无误差文字)
        plot_3d_poses(rvecs, tvecs, PATTERN_SIZE, SQUARE_SIZE_MM, err)
        
        print("\nDone. Please manually save the figures from the popup windows.")
    else:
        print("Calibration failed.")