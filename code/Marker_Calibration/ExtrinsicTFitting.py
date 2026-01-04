import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # 新增：用于绘制相机视锥
import os

# --- 0. 全局绘图设置 (SCI 风格) ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'          # 全局字体加粗
plt.rcParams['axes.labelweight'] = 'bold'     # 坐标轴标签加粗
plt.rcParams['axes.titleweight'] = 'bold'     # 标题加粗
plt.rcParams['axes.labelsize'] = 12           # 坐标轴字体大小
plt.rcParams['font.size'] = 12                # 基础字体大小
plt.rcParams['legend.fontsize'] = 10          # 图例字体大小
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['savefig.dpi'] = 300             # 保存分辨率

# --- 从Excel读取内参 ---
def load_intrinsics_from_excel(filepath):
    """
    从指定的Excel文件中加载相机内参和畸变系数。
    Excel文件应包含'Parameter'和'Value'列。
    """
    print(f"Attempting to load intrinsic parameters from '{filepath}'...")
    try:
        df = pd.read_excel(filepath)
        # 使用'Parameter'列作为索引，方便快速查找
        params = df.set_index('Parameter')['Value']

        # 构建相机内参矩阵
        camera_matrix = np.array([
            [params['fx'], params['skew'], params['cx']],
            [0,            params['fy'],   params['cy']],
            [0,            0,              1]
        ], dtype=np.float32)

        # 构建畸变系数数组
        dist_keys = ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6']
        dist_coeffs_list = [params[key] for key in dist_keys if key in params.index]
        dist_coeffs = np.array(dist_coeffs_list, dtype=np.float32).reshape(1, -1)

        print("Successfully loaded intrinsic parameters.")
        return camera_matrix, dist_coeffs

    except FileNotFoundError:
        print(f"ERROR: Intrinsic parameters file not found at '{filepath}'.")
        print("Please run the intrinsic calibration script first.")
        return None, None
    except KeyError as e:
        print(f"ERROR: The Excel file is missing a required parameter: {e}.")
        return None, None
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while loading intrinsics: {e}")
        return None, None

# --- REWRITTEN FUNCTION: 将外参保存到Excel ---
def save_extrinsics_to_excel(R_wc, T_wc, error, filepath):
    """
    将外参（世界到相机，相机到世界）和重投影误差保存到Excel文件。
    格式与内参文件保持一致。
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_rows = []

    # --- 1. 世界到相机的变换 (World -> Camera) ---
    data_rows.append(['--- World to Camera Transformation ---', '', ''])
    # 旋转矩阵 R_wc
    for i in range(3):
        for j in range(3):
            data_rows.append([f'R_wc_{i+1}{j+1}', R_wc[i, j], f'Rotation matrix element ({i+1},{j+1})'])
    # 平移向量 T_wc
    T_wc_flat = T_wc.flatten()
    data_rows.append(['Tx_wc', T_wc_flat[0], 'Translation on X-axis (mm)'])
    data_rows.append(['Ty_wc', T_wc_flat[1], 'Translation on Y-axis (mm)'])
    data_rows.append(['Tz_wc', T_wc_flat[2], 'Translation on Z-axis (mm)'])
    data_rows.append(['---', '---', '---'])

    # --- 2. 相机到世界的变换 (Camera -> World) ---
    R_cw = R_wc.T
    T_cw = -R_cw @ T_wc
    data_rows.append(['--- Camera to World Transformation ---', '', ''])
    # 旋转矩阵 R_cw
    for i in range(3):
        for j in range(3):
            data_rows.append([f'R_cw_{i+1}{j+1}', R_cw[i, j], f'Rotation matrix element ({i+1},{j+1})'])
    # 平移向量 T_cw
    T_cw_flat = T_cw.flatten()
    data_rows.append(['Tx_cw', T_cw_flat[0], 'Camera position on X-axis in World (mm)'])
    data_rows.append(['Ty_cw', T_cw_flat[1], 'Camera position on Y-axis in World (mm)'])
    data_rows.append(['Tz_cw', T_cw_flat[2], 'Camera position on Z-axis in World (mm)'])
    data_rows.append(['---', '---', '---'])

    # --- 3. 标定质量 ---
    data_rows.append(['--- Calibration Quality ---', '', ''])
    data_rows.append(['Reprojection Error', error, 'Mean error in pixels'])

    # 创建DataFrame并保存
    df = pd.DataFrame(data_rows, columns=['Parameter', 'Value', 'Description'])
    try:
        df.to_excel(filepath, index=False, engine='openpyxl')
        print(f"\nExtrinsic parameters successfully saved to: '{filepath}'")
    except Exception as e:
        print(f"\nERROR: Failed to save extrinsic parameters to Excel. Reason: {e}")


def calibrate_camera_extrinsics(object_points_world, image_points_pixel, camera_matrix, dist_coeffs):
    # (此函数内容保持不变)
    if object_points_world.ndim == 2:
        object_points_world = object_points_world.reshape(-1, 1, 3).astype(np.float32)
    elif object_points_world.ndim == 3 and object_points_world.shape[1] != 1:
        object_points_world = object_points_world.reshape(-1, 1, 3).astype(np.float32)
    else:
        object_points_world = object_points_world.astype(np.float32)

    if image_points_pixel.ndim == 2:
        image_points_pixel = image_points_pixel.reshape(-1, 1, 2).astype(np.float32)
    elif image_points_pixel.ndim == 3 and image_points_pixel.shape[1] != 1:
        image_points_pixel = image_points_pixel.reshape(-1, 1, 2).astype(np.float32)
    else:
        image_points_pixel = image_points_pixel.astype(np.float32)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points_world, 
        image_points_pixel, 
        camera_matrix, 
        dist_coeffs,
        iterationsCount=100,
        reprojectionError=8.0,
        confidence=0.99,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if success:
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        projected_image_points, _ = cv2.projectPoints(object_points_world, rvec, tvec, camera_matrix, dist_coeffs)
        
        error_per_point = np.sqrt(np.sum((np.squeeze(image_points_pixel) - np.squeeze(projected_image_points))**2, axis=1))
        mean_error = np.mean(error_per_point)
        
        print(f"PnP solved successfully. Number of inliers: {len(inliers) if inliers is not None else 'N/A'}")
        print(f"Mean reprojection error (NumPy calculated): {mean_error:.4f} pixels")
        
        return rotation_matrix, tvec, mean_error
    else:
        print("Error: solvePnP/solvePnPRansac failed to find a solution.")
        return None, None, None

def load_world_coordinates_from_csv(csv_filepath, id_col='marker_id', x_col='Xw', y_col='Yw', z_col='Zw'):
    # (此函数内容保持不变)
    try:
        df = pd.read_csv(csv_filepath)
        if not all(col in df.columns for col in [id_col, x_col, y_col, z_col]):
            print(f"Error: CSV '{csv_filepath}' is missing one of required columns: {id_col}, {x_col}, {y_col}, {z_col}")
            return None
        world_coords_map = {}
        for _, row in df.iterrows():
            world_coords_map[str(row[id_col])] = np.array([row[x_col], row[y_col], row[z_col]], dtype=np.float32)
        return world_coords_map
    except FileNotFoundError:
        print(f"Error: Input file not found at '{csv_filepath}'.")
        print("Please ensure the file exists in the specified subfolder.")
        return None
    except Exception as e:
        print(f"Error loading world coordinates from CSV '{csv_filepath}': {e}")
        return None

def load_image_coordinates_from_csv(csv_filepath, id_col='marker_id', u_col='u', v_col='v'):
    # (此函数内容保持不变)
    try:
        df = pd.read_csv(csv_filepath)
        if not all(col in df.columns for col in [id_col, u_col, v_col]):
            print(f"Error: CSV '{csv_filepath}' is missing one of required columns: {id_col}, {u_col}, {v_col}")
            return None
        image_coords_map = {}
        for _, row in df.iterrows():
            image_coords_map[str(row[id_col])] = np.array([row[u_col], row[v_col]], dtype=np.float32)
        return image_coords_map
    except FileNotFoundError:
        print(f"Error: Input file not found at '{csv_filepath}'.")
        print("Please ensure the file exists in the specified subfolder.")
        return None
    except Exception as e:
        print(f"Error loading image coordinates from CSV '{csv_filepath}': {e}")
        return None

def save_camera_coordinates(world_points, marker_ids, R_world_to_cam, T_world_to_cam, 
                            camera_matrix, real_marker_diameter_mm, filepath):
    # (此函数内容保持不变)
    if world_points.ndim != 2 or world_points.shape[1] != 3:
        print("Error: world_points must be a Nx3 array.")
        return

    camera_points_list = []
    for point_w in world_points:
        point_c = (R_world_to_cam @ point_w.reshape(3, 1) + T_world_to_cam).flatten()
        camera_points_list.append(point_c)
    
    camera_points_array = np.array(camera_points_list)

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    f_avg_pixels = (fx + fy) / 2.0

    zc_depths = camera_points_array[:, 2]

    theoretical_major_axis = np.zeros_like(zc_depths)
    valid_depths_mask = zc_depths > 1e-6 
    theoretical_major_axis[valid_depths_mask] = (f_avg_pixels * real_marker_diameter_mm) / zc_depths[valid_depths_mask]
    
    df_camera_coords = pd.DataFrame({
        'marker_id': marker_ids,
        'Xc': camera_points_array[:, 0],
        'Yc': camera_points_array[:, 1],
        'Zc': camera_points_array[:, 2],
        'major_axis': theoretical_major_axis
    })

    try:
        df_camera_coords.rename(columns={'Xc': 'Xw', 'Yc': 'Yw', 'Zc': 'Zw'}, inplace=True)
        df_camera_coords.to_csv(filepath, index=False, float_format='%.8f')
        print(f"\nSuccessfully saved camera coordinates and theoretical major_axis to '{filepath}'")
    except Exception as e:
        print(f"\nError: Failed to save camera coordinates to '{filepath}'. Reason: {e}")

# --- 【核心修改】SCI 标准绘图函数 ---
def plot_3d_comparison(world_points, image_points_for_pnp, R_world_to_cam, T_world_to_cam, camera_matrix, title_suffix=""):
    """
    绘制世界坐标系下的场景：
    1. 原始世界点 (World Points)
    2. 相机姿态 (Camera Pose) - 使用视锥体表示
    """
    if world_points is None or R_world_to_cam is None or T_world_to_cam is None:
        print("Cannot plot, missing world points or R, T.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # --- 1. 绘制世界坐标系下的点 (Ground Truth) ---
    wp_plot = world_points.reshape(-1, 3)
    ax.scatter(wp_plot[:, 0], wp_plot[:, 1], wp_plot[:, 2], 
               c='steelblue', marker='o', s=30, alpha=0.8, label='World Points ($P_w$)')

    # --- 2. 计算相机在世界坐标系下的位置和姿态 ---
    # 公式: P_c = R * P_w + T  =>  P_w = R^T * (P_c - T)
    # 相机原点在相机坐标系下是 (0,0,0)，转换到世界坐标系:
    R_cam_to_world = R_world_to_cam.T
    cam_origin_in_world = -R_cam_to_world @ T_world_to_cam
    
    # --- 3. 绘制相机视锥体 (Pyramid) ---
    # 定义相机坐标系下的标准视锥体 (指向 Z 轴正方向)
    scale = 10.0 # 视锥体大小，根据数据范围调整
    # 顶点: 镜头中心(0,0,0) 和 底面四个点
    cam_corners_c = np.array([
        [0, 0, 0],              # Center
        [-scale, -scale, scale], # Top-Left
        [scale, -scale, scale],  # Top-Right
        [scale, scale, scale],   # Bottom-Right
        [-scale, scale, scale]   # Bottom-Left
    ])
    
    # 将视锥体变换到世界坐标系
    cam_corners_w = []
    for pt in cam_corners_c:
        # P_w = R^T * P_c + C_w (其中 C_w 是相机在世界的位置)
        # 或者直接使用逆变换: P_w = R^T * (P_c - T) -> 这里 P_c 是点，T 是平移
        # 更简单的理解: 旋转点 + 移动到相机位置
        pt_w = R_cam_to_world @ pt.reshape(3, 1) + cam_origin_in_world
        cam_corners_w.append(pt_w.flatten())
    cam_corners_w = np.array(cam_corners_w)
    
    # 定义面
    verts = [
        [cam_corners_w[0], cam_corners_w[1], cam_corners_w[2]], # Side 1
        [cam_corners_w[0], cam_corners_w[2], cam_corners_w[3]], # Side 2
        [cam_corners_w[0], cam_corners_w[3], cam_corners_w[4]], # Side 3
        [cam_corners_w[0], cam_corners_w[4], cam_corners_w[1]], # Side 4
        [cam_corners_w[1], cam_corners_w[2], cam_corners_w[3], cam_corners_w[4]] # Base
    ]
    
    # 绘制半透明红色视锥
    ax.add_collection3d(Poly3DCollection(verts, facecolors='crimson', linewidths=1, edgecolors='darkred', alpha=0.35))
    
    # 标记相机中心
    ax.scatter(cam_origin_in_world[0], cam_origin_in_world[1], cam_origin_in_world[2],
               c='red', marker='s', s=50, label='Camera Pose ($C_w$)')

    # --- 4. 绘制世界坐标原点 ---
    ax.scatter(0, 0, 0, c='black', marker='x', s=60, label='World Origin (0,0,0)')

    # --- 5. 美化设置 (SCI 风格) ---
    ax.set_xlabel('X (mm)', labelpad=10, fontweight='bold')
    ax.set_ylabel('Y (mm)', labelpad=10, fontweight='bold')
    ax.set_zlabel('Z (mm)', labelpad=10, fontweight='bold')
    ax.set_title(f'Extrinsic Calibration Result\n(World Frame Visualization)', fontsize=14, pad=15)
    
    # 去除背景网格和灰色填充，实现干净的白色背景
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    # 设置图例
    ax.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9)
    
    # --- 6. 关键：设置等比例坐标轴 (Equal Aspect Ratio) ---
    # 这一步防止 3D 空间变形
    all_points = np.vstack([wp_plot, cam_corners_w])
    X, Y, Z = all_points[:,0], all_points[:,1], all_points[:,2]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x, mid_y, mid_z = (X.max()+X.min())*0.5, (Y.max()+Y.min())*0.5, (Z.max()+Z.min())*0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 调整视角
    ax.view_init(elev=30, azim=135)
    plt.tight_layout()
    plt.show()


# 提取未畸变矫正的内参执行pnp算法的程序主模块
if __name__ == "__main__":
    # --- 1. Define File Paths ---
    data_folder_path = os.path.join("Results", "data", 'PreprocessPara')
    os.makedirs(data_folder_path, exist_ok=True)
    print(f"Reading/Writing files from/to: '{data_folder_path}'")

    intrinsic_params_path = os.path.join(data_folder_path, 'IntrinsicParameters.xlsx')
    extrinsic_params_path = os.path.join(data_folder_path, 'ExtrinsicParameters.xlsx')
    
    # world_csv_path = os.path.join(data_folder_path, 'world_marker.csv')
    world_csv_path = os.path.join(data_folder_path, 'world_marker_CMM.csv')
    pixel_csv_path = os.path.join(data_folder_path, 'pixel_marker.csv')
    
    # marker_real_diameter = 3.0 # 5*5阵列marker直径
    marker_real_diameter = 2.0 # circle阵列marker直径

    # --- 2. Load Intrinsic Parameters from Excel ---
    camera_intrinsic_matrix, distortion_coefficients = load_intrinsics_from_excel(intrinsic_params_path)
    
    if camera_intrinsic_matrix is None:
        print("Exiting due to failure in loading intrinsic parameters.")
        exit()

    # --- 3. Load 3D-2D correspondences ---
    world_coords_map = load_world_coordinates_from_csv(world_csv_path)
    image_coords_map = load_image_coordinates_from_csv(pixel_csv_path)

    if world_coords_map is None or image_coords_map is None:
        print("Failed to load coordinate data. Exiting.")
        exit()
    if not world_coords_map or not image_coords_map:
        print("Coordinate maps are empty after loading. Exiting.")
        exit()

    common_ids_world = {str(k) for k in world_coords_map.keys()}
    common_ids_image = {str(k) for k in image_coords_map.keys()}
    common_ids = sorted(list(common_ids_world & common_ids_image), key=lambda x: int(float(x)))

    if not common_ids:
        print("No common marker IDs found between world and image coordinate files.")
        exit()
    print(f"Found {len(common_ids)} common marker IDs for PnP: {common_ids[:10]}...")

    object_points_3d_world_list = [world_coords_map[id_val] for id_val in common_ids]
    image_points_2d_pixel_list = [image_coords_map[id_val] for id_val in common_ids]
    
    object_points_3d_world_for_pnp = np.array(object_points_3d_world_list, dtype=np.float32)
    image_points_2d_pixel_for_pnp = np.array(image_points_2d_pixel_list, dtype=np.float32)

    # --- 4. Calculate Extrinsic Parameters ---
    R, T, reproj_err = calibrate_camera_extrinsics(
        object_points_3d_world_for_pnp,
        image_points_2d_pixel_for_pnp,
        camera_intrinsic_matrix,
        distortion_coefficients
    )

    # --- 5. Process and Save Results ---
    if R is not None and T is not None:
        print("\n--- Camera Extrinsic Calibration Results ---")
        print("Rotation Matrix (R) - World to Camera:")
        print(R)
        print("\nTranslation Vector (T) - World to Camera (in units of world coordinates):")
        print(T)
        print(f"\nMean Reprojection Error: {reproj_err:.4f} pixels")
        
        # Save results to the new Excel file
        save_extrinsics_to_excel(R, T, reproj_err, extrinsic_params_path)

        # The rest of the logic remains the same
        output_camera_coords_path = os.path.join(data_folder_path, 'camera_marker.csv')
        
        save_camera_coordinates(
            object_points_3d_world_for_pnp,
            common_ids,
            R,
            T,
            camera_intrinsic_matrix,
            marker_real_diameter,
            output_camera_coords_path
        )

        plot_3d_comparison(
            object_points_3d_world_for_pnp,
            image_points_2d_pixel_for_pnp,
            R,
            T,
            camera_intrinsic_matrix,
            title_suffix="(PnP Result)"
        )
    else:
        print("\nCamera extrinsic calibration failed.")