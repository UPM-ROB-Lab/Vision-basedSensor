import cv2
import numpy as np
import os
import pandas as pd

# def cresize_crop_mini(img, crop_ratios=(1/8, 1/8, 0, 1/16)): # 5*5阵列裁剪系数
def cresize_crop_mini(img, crop_ratios=(1/8, 1/8, 1/16, 0)): # 环形阵列裁剪系数
    """
    根据给定的比例裁剪图像的左侧、右侧、顶部和底部。
    Args:
        img (np.ndarray): 输入的图像。
        crop_ratios (tuple): 四个方向的裁剪比例 (左, 右, 上, 下)。
    Returns:
        tuple: (裁剪后的图像, 新的宽度, 新的高度)
    """
    h_original, w_original = img.shape[:2]
    
    # 计算需要裁剪的像素数
    left = int(w_original * crop_ratios[0])
    right = int(w_original * crop_ratios[1])
    top = int(h_original * crop_ratios[2])
    bottom = int(h_original * crop_ratios[3])
    
    # 定义裁剪区域的起始和结束索引
    start_row = top
    end_row = h_original - bottom
    start_col = left
    end_col = w_original - right
    
    # 执行裁剪
    cropped_img = img[start_row:end_row, start_col:end_col]
    
    # 获取新的尺寸
    new_h, new_w = cropped_img.shape[:2]
    return cropped_img, new_w, new_h

def save_results_to_excel(camera_matrix, dist_coeffs, reproj_error, filepath):
    """
    将相机标定结果保存到单个工作表的Excel文件中。
    Args:
        camera_matrix (np.ndarray): 3x3 的相机内参矩阵。
        dist_coeffs (np.ndarray): 1xN 的畸变系数。
        reproj_error (float): 标定的平均重投影误差。
        filepath (str): 输出的Excel文件路径。
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: '{output_dir}'")
    # 准备要写入Excel的数据
    data_rows = []
    # 添加内参矩阵的每一项
    data_rows.append(['fx', camera_matrix[0, 0], 'Focal length in x-axis (pixels)'])
    data_rows.append(['fy', camera_matrix[1, 1], 'Focal length in y-axis (pixels)'])
    data_rows.append(['cx', camera_matrix[0, 2], 'Principal point x-coordinate (pixels)'])
    data_rows.append(['cy', camera_matrix[1, 2], 'Principal point y-coordinate (pixels)'])
    data_rows.append(['skew', camera_matrix[0, 1], 'Skew coefficient'])
    
    # 添加一个空行作为分隔
    data_rows.append(['---', '---', '---'])
    # 添加畸变系数
    dist_labels = ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6']
    dist_descriptions = [
        'Radial distortion coefficient', 'Radial distortion coefficient',
        'Tangential distortion coefficient', 'Tangential distortion coefficient',
        'Radial distortion coefficient', 'Radial distortion coefficient',
        'Radial distortion coefficient', 'Radial distortion coefficient'
    ]
    
    for i, coeff in enumerate(dist_coeffs.flatten()):
        if i < len(dist_labels):
            data_rows.append([dist_labels[i], coeff, dist_descriptions[i]])
        else:
            data_rows.append([f'dist_coeff_{i+1}', coeff, 'Additional distortion coefficient'])
    # 添加一个空行作为分隔
    data_rows.append(['---', '---', '---'])
    # 添加重投影误差
    data_rows.append(['Reprojection Error', reproj_error, 'Mean error in pixels'])
    # 创建DataFrame
    df = pd.DataFrame(data_rows, columns=['Parameter', 'Value', 'Description'])
    try:
        # 将DataFrame保存到Excel
        df.to_excel(filepath, index=False, engine='openpyxl')
        print(f"\nCalibration results successfully saved to: '{filepath}'")
    except Exception as e:
        print(f"\nERROR: Failed to save results to Excel file. Reason: {e}")

def calibrate_from_images(image_folder, pattern_size, square_size_mm, show_corners=True):
    """
    使用张正友标定法从文件夹中的图像进行相机标定。
    (修改版：在处理前会先对图像进行裁剪)
    Args:
        image_folder (str): 存放标定图像的文件夹路径。
        pattern_size (tuple): 棋盘格内部角点的数量 (列数, 行数)。例如 (9, 6)。
        square_size_mm (float): 棋盘格单个方块的边长（单位：毫米）。
        show_corners (bool): 是否显示检测到的角点图像。
    Returns:
        tuple: (camera_matrix, dist_coeffs, reproj_error) 如果成功，否则 (None, None, None)。
    """
    print("--- Starting Camera Calibration ---")
    print(f"Chessboard inner corner size: {pattern_size}")
    print(f"Chessboard square size: {square_size_mm} mm")
    print("Note: Each image will be cropped before processing.")

    # 准备3D世界坐标点 (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp = objp * square_size_mm  # 缩放到真实世界尺寸
    # 用于存储所有图像的3D点和2D点
    obj_points = []  # 3D 世界坐标点
    img_points = []  # 2D 图像坐标点
    # 获取文件夹中所有图像的路径
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print(f"ERROR: No image files found in folder '{image_folder}'.")
        return None, None, None
    print(f"\nFound {len(image_files)} images, starting processing...")
    
    # ### MODIFIED ###: image_size 将在处理第一张图后根据裁剪后的尺寸确定
    image_size = None
    found_count = 0
    for fname in image_files:
        img_original = cv2.imread(fname)
        if img_original is None:
            print(f"Warning: Could not read image {fname}, skipping.")
            continue
        
        # ### MODIFIED ###: 使用新的裁剪格式 (左, 右, 上, 下)
        # img, new_w, new_h = cresize_crop_mini(img_original, crop_ratios=(1/8, 1/8, 0, 1/16))
        img, new_w, new_h = cresize_crop_mini(img_original, crop_ratios=(1/8, 1/8, 1/16, 0))
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # ### MODIFIED ###: 使用裁剪后的图像尺寸
        if image_size is None:
            image_size = (new_w, new_h)
            print(f"Images will be processed at cropped resolution: {image_size[0]}x{image_size[1]}")
        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        # 如果找到角点
        if ret:
            found_count += 1
            print(f"  - {os.path.basename(fname)}: Cropped and Corners found successfully.")
            obj_points.append(objp)
            # 提高角点检测精度（亚像素级别）
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners2)
            # 可选：绘制并显示角点 (在裁剪后的图像上)
            if show_corners:
                cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
                # 缩小图像以便显示
                display_img = cv2.resize(img, (image_size[0] // 2, image_size[1] // 2))
                cv2.imshow('Detected Corners (on Cropped Image)', display_img)
                cv2.waitKey(500) # 等待0.5秒
        else:
            print(f"  - {os.path.basename(fname)}: Cropped, but Corners not found.")
    if show_corners:
        cv2.destroyAllWindows()
    if found_count < 5:
        print(f"\nCalibration failed: Only found corners in {found_count} images. At least 5 are required.")
        return None, None, None
    print(f"\nSuccessfully detected corners in {found_count}/{len(image_files)} images. Calculating camera parameters...")
    # --- 核心步骤：执行相机标定 ---
    # ret: 总体重投影误差
    # mtx: 相机内参矩阵
    # dist: 畸变系数
    # rvecs: 旋转向量 (每个图像一个)
    # tvecs: 平移向量 (每个图像一个)
    # ### MODIFIED ###: 确保使用正确的 image_size (裁剪后的尺寸)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)
    if ret:
        print("\n--- Calibration Complete ---")
        return mtx, dist, ret
    else:
        print("\nCalibration failed: cv2.calibrateCamera function did not return a valid result.")
        return None, None, None

if __name__ == "__main__":
    # --- 1. 用户配置区 ---
    # 存放标定图像的文件夹名称
    IMAGE_FOLDER = 'calibration_images'
    
    # 棋盘格的内部角点数量 (宽, 高) 或 (列, 行)
    PATTERN_SIZE = (6, 6)
    
    # 棋盘格中每个方块的实际边长（单位：毫米）
    SQUARE_SIZE_MM = 3.0
    
    # 输出的Excel文件名和路径
    OUTPUT_FILE = os.path.join('Results', 'data', 'PreprocessPara', 'IntrinsicParameters640_3.xlsx')
    
    # 是否在标定过程中显示检测到的角点图像
    SHOW_CORNERS_VISUALIZATION = True
    
    # --- 2. 检查并创建文件夹 ---
    if not os.path.isdir(IMAGE_FOLDER):
        print(f"Folder '{IMAGE_FOLDER}' does not exist. Creating it...")
        os.makedirs(IMAGE_FOLDER)
        print(f"Folder created. Please place your chessboard images in '{IMAGE_FOLDER}' and run the script again.")
        exit()
    # --- 3. 执行标定 ---
    camera_matrix, dist_coeffs, reproj_error = calibrate_from_images(
        image_folder=IMAGE_FOLDER,
        pattern_size=PATTERN_SIZE,
        square_size_mm=SQUARE_SIZE_MM,
        show_corners=SHOW_CORNERS_VISUALIZATION
    )
    # --- 4. 处理并保存结果 ---
    if camera_matrix is not None and dist_coeffs is not None:
        print("\n--- Calibration Results Summary ---")
        print(f"Mean Reprojection Error: {reproj_error:.4f} pixels")
        print("\nCamera Matrix (K):")
        print(camera_matrix)
        print("\nDistortion Coefficients (k1, k2, p1, p2, k3,...):")
        print(dist_coeffs)
        
        # 将结果保存到Excel文件
        save_results_to_excel(camera_matrix, dist_coeffs, reproj_error, OUTPUT_FILE)
    else:
        print("\nCalibration process could not be completed. Please check your images or configuration.")