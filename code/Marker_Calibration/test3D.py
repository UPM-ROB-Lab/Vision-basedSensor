import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter

# ------------------------------
# 全局配置参数
# ------------------------------
# REAL_MARKER_DIAMETER_MM = 3.0 # 5*5阵列marker直径
REAL_MARKER_DIAMETER_MM = 2.0 # circle阵列marker直径

# =============================================================================
# 核心辅助函数 (加载参数和坐标转换)
# (这部分函数保持不变)
# =============================================================================

def load_intrinsics_from_excel(filepath):
    """从指定的Excel文件加载相机内参矩阵和畸变系数。"""
    try:
        df = pd.read_excel(filepath)
        params = df.set_index('Parameter')['Value']
        camera_matrix = np.array([
            [params['fx'], params.get('skew', 0), params['cx']],
            [0,             params['fy'],          params['cy']],
            [0,             0,                     1]
        ], dtype=np.float32)
        dist_keys = ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6']
        dist_coeffs_list = [params[key] for key in dist_keys if key in params.index]
        dist_coeffs = np.array(dist_coeffs_list, dtype=np.float32)
        return camera_matrix, dist_coeffs
    except Exception as e:
        print(f"--> 错误: 加载内参文件 '{filepath}' 失败: {e}")
        return None, None

def load_extrinsics_from_excel(filepath):
    """从指定的Excel文件加载旋转矩阵(R)和平移向量(T)。"""
    try:
        df = pd.read_excel(filepath)
        params = df.set_index('Parameter')['Value']
        R_world_to_cam = np.array([
            [params['R_wc_11'], params['R_wc_12'], params['R_wc_13']],
            [params['R_wc_21'], params['R_wc_22'], params['R_wc_23']],
            [params['R_wc_31'], params['R_wc_32'], params['R_wc_33']]
        ], dtype=np.float32)
        T_world_to_cam = np.array([
            params['Tx_wc'], params['Ty_wc'], params['Tz_wc']
        ], dtype=np.float32).reshape(3, 1)
        return R_world_to_cam, T_world_to_cam
    except Exception as e:
        print(f"--> 错误: 加载外参文件 '{filepath}' 失败: {e}")
        return None, None

def calculate_3d_point_twostep(pixel_coord, pixel_major_axis, camera_matrix, dist_coeffs):
    """使用两步法，将2D像素信息反解为3D相机坐标。"""
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    f_avg = (fx + fy) / 2.0
    pixel_reshaped = np.array([[pixel_coord]], dtype=np.float32)
    undistorted_pixel = cv2.undistortPoints(pixel_reshaped, camera_matrix, dist_coeffs, None, camera_matrix)[0,0]
    u_undist, v_undist = undistorted_pixel[0], undistorted_pixel[1]
    
    # 距离 R from center
    R_px = np.sqrt((u_undist - cx)**2 + (v_undist - cy)**2)
    
    # R: Marker半径在焦平面上的投影半径
    # 根据相似三角形原理，d/R = D/Zc => Zc = D*R/d 
    # 但是我们现在没有 d，只有像素长轴 `pixel_major_axis` 
    # 假设 d_effective = D * f_avg / R_px (近似公式，用于估计深度)
    # d_effective = (REAL_MARKER_DIAMETER_MM / np.sqrt(R_px**2 + f_avg**2)) * f_avg
    d_effective = (REAL_MARKER_DIAMETER_MM / f_avg) * np.sqrt(R_px**2 + f_avg**2)
    
    # 根据像素长轴和有效直径计算 Zc
    if pixel_major_axis < 1e-6: return None
    Zc = f_avg * (d_effective / pixel_major_axis)
    
    # 根据 Zc 计算 Xc 和 Yc
    Xc = Zc * (u_undist - cx) / fx
    Yc = Zc * (v_undist - cy) / fy
    return np.array([Xc, Yc, Zc])

# =============================================================================
# 主程序
# =============================================================================

def visualize_and_compare_with_alignment():
    """执行反解、对齐、计算误差，并分别在两张独立的图中进行可视化。"""
    
    # --- 0. 定义文件路径 (假设文件路径已在运行环境中存在) ---
    base_dir = os.path.join("Results", "data", "PreprocessPara")
    raw_marker_txt = os.path.join(base_dir, 'MarkerCalibration.txt')
    intrinsic_params_xlsx = os.path.join(base_dir, "IntrinsicParameters.xlsx")
    extrinsic_params_xlsx = os.path.join(base_dir, "ExtrinsicParameters.xlsx")
    ground_truth_csv = os.path.join(base_dir, "world_marker_CMM.csv")

    # --- 1. 加载数据和参数 ---
    print("--- 步骤 1/6: 加载数据和参数 ---")
    try:
        df = pd.read_csv(raw_marker_txt, header=None, skiprows=1, sep='\s+', usecols=[0, 1, 2, 5, 6, 7],
                             names=['frameno', 'row', 'col', 'u', 'v', 'major_axis'], engine='python')
        for col in df.columns:
            if df[col].dtype == 'object': df[col] = df[col].str.strip(',')
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        grouped = df.groupby(['row', 'col'])
        pixel_df = grouped[['u', 'v', 'major_axis']].mean().reset_index()
        pixel_df.sort_values(by=['row', 'col'], inplace=True)
        pixel_df['marker_id'] = range(1, len(pixel_df) + 1)
        camera_matrix, dist_coeffs = load_intrinsics_from_excel(intrinsic_params_xlsx)
        R_wc, T_wc = load_extrinsics_from_excel(extrinsic_params_xlsx)
        truth_df = pd.read_csv(ground_truth_csv)
        truth_df.drop_duplicates(subset='marker_id', keep='first', inplace=True)
        truth_df.rename(columns={'Xw': 'Xw_truth', 'Yw': 'Yw_truth', 'Zw': 'Zw_truth'}, inplace=True)
        if camera_matrix is None or R_wc is None: return
        print(f"-> 成功加载了 {len(pixel_df)} 个标记点, 相机参数和 {len(truth_df)} 个真值点。")
    except Exception as e:
        print(f"--> 错误: 数据加载或处理失败: {e}"); return

    # --- 2. 反解3D坐标 ---
    print("\n--- 步骤 2/6: 反解3D坐标 ---")
    world_coords_list = [
        {'marker_id': int(row['marker_id']), **dict(zip(['Xw', 'Yw', 'Zw'], (R_wc.T @ (calculate_3d_point_twostep((row['u'], row['v']), row['major_axis'], camera_matrix, dist_coeffs).reshape(3, 1) - T_wc)).flatten()))}
        for _, row in pixel_df.iterrows()
        if calculate_3d_point_twostep((row['u'], row['v']), row['major_axis'], camera_matrix, dist_coeffs) is not None
    ]
    calculated_df = pd.DataFrame(world_coords_list)
    print(f"-> 成功计算了 {len(calculated_df)} 个点的3D世界坐标。")
    
    # --- 3. 计算最优平移和绝对误差 ---
    print("\n--- 步骤 3/6: 计算最优平移和绝对误差 ---")
    merged_df = pd.merge(calculated_df, truth_df, on='marker_id')
    if merged_df.empty: print("--> 错误: 计算结果与真值没有共同的 marker_id。"); return

    calc_points = merged_df[['Xw', 'Yw', 'Zw']].values
    truth_points = merged_df[['Xw_truth', 'Yw_truth', 'Zw_truth']].values
    
    # 计算质心
    centroid_calc = np.mean(calc_points, axis=0)
    centroid_truth = np.mean(truth_points, axis=0)
    # 计算最优平移向量
    optimal_shift = centroid_calc - centroid_truth
    
    # 将真值点平移到与计算点对齐
    shifted_truth_points = truth_points + optimal_shift
    error_vectors = calc_points - shifted_truth_points
    
    # 计算误差
    merged_df['error_x'] = error_vectors[:, 0]
    merged_df['error_y'] = error_vectors[:, 1]
    merged_df['error_z'] = error_vectors[:, 2]
    merged_df['error_x_abs'] = np.abs(error_vectors[:, 0])
    merged_df['error_y_abs'] = np.abs(error_vectors[:, 1])
    merged_df['error_z_abs'] = np.abs(error_vectors[:, 2])
    merged_df['error_total'] = np.linalg.norm(error_vectors, axis=1)
    
    average_error_total = merged_df['error_total'].mean()
    average_error_x = merged_df['error_x_abs'].mean()
    average_error_y = merged_df['error_y_abs'].mean()
    average_error_z = merged_df['error_z_abs'].mean()
    print("-> 误差计算完成。")

    # --- 4. 创建 SCI 风格的 3D 位姿对比图 (图 1) ---
    print("\n--- 步骤 4/6: 创建 SCI 风格的 3D 位姿对比图 ---")
    
    # 使用白色背景和 seaborn-v0_8-whitegrid 风格
    plt.style.use('seaborn-v0_8-whitegrid')
    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    # 1. 绘制 Calculated Pose (蓝色点，高透明度)
    # 使用较低的 alpha 值实现透明，使背后的误差连线和真值点更容易被看到
    # 使用 'o' marker，并设置边框
    ax_3d.scatter(
        merged_df['Xw'], merged_df['Yw'], merged_df['Zw'], 
        c='blue', marker='o', s=80, edgecolors='k', linewidths=0.5, alpha=0.4, 
        label='Calculated Pose' # 降低透明度 (alpha=0.4)
    )
    
    # 2. 绘制 Ground Truth (对齐后的绿色点，高对比度)
    # 使用星形 'P' marker，s 稍大，保证清晰可见
    ax_3d.scatter(
        shifted_truth_points[:, 0], shifted_truth_points[:, 1], shifted_truth_points[:, 2], 
        c='green', marker='P', s=120, alpha=1.0, 
        label='Ground Truth (Aligned)'
    )
    
    # 3. 绘制误差连线 (强调误差，使用醒目的颜色和粗线)
    # 使用红色连线 ('r') 并增加线宽 (linewidth=1.5)，使误差向量更明显
    for i in range(len(merged_df)):
        p_calc = calc_points[i]
        p_shifted = shifted_truth_points[i]
        
        # 绘制误差向量
        ax_3d.plot(
            [p_calc[0], p_shifted[0]], 
            [p_calc[1], p_shifted[1]], 
            [p_calc[2], p_shifted[2]], 
            color='red', linestyle='-', linewidth=1.5, alpha=0.7 # 误差连线加粗，并使用红色
        )
        
        # 在真值点附近标注 Marker ID (字体放大到 11)
        ax_3d.text(p_shifted[0], p_shifted[1], p_shifted[2] + merged_df['error_total'].max() * 0.1, 
                   f" {merged_df.iloc[i]['marker_id']}", color='black', fontsize=11, weight='bold')
    
    # 4. 确保等比例尺 (防止失真，这是 SCI 3D 图的关键)
    def set_axes_equal(ax):
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
        ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
        ax.set_zlim3d([origin[2] - radius, origin[2] + radius])
        
    set_axes_equal(ax_3d) # 应用等比例尺
    
    # 5. 设置专业化标签和标题 (字体放大)
    ax_3d.set_xlabel('World X (mm)', fontsize=14); ax_3d.set_ylabel('World Y (mm)', fontsize=14); ax_3d.set_zlabel('World Z (mm)', fontsize=14)
    ax_3d.set_title(f'3D Pose Comparison (Mean Total Error: {average_error_total:.4f} mm)', fontsize=18, weight='bold')
    ax_3d.legend(loc='best', fontsize=12)
    ax_3d.grid(False) # <--- **修改：移除 3D 图的网格线**
    ax_3d.tick_params(labelsize=12)
    fig_3d.tight_layout()

    # --- 5. 创建 2x2 误差分析图 (图 2, 字体放大) ---
    print("\n-> 配置第二张图: 2x2误差分析图 (风格优化)...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig_error, axs = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
    fig_error.suptitle('Per-Marker Absolute Error Analysis', fontsize=24, weight='bold') # 放大到 24
    marker_ids = merged_df['marker_id']

    # 绘制 X, Y, Z, Total 误差
    error_data = [
        (merged_df['error_total'], 'Total Distance Error', 'black', 'P-', average_error_total),
        (merged_df['error_x_abs'], 'Absolute X-axis Error', 'red', 'o--', average_error_x),
        (merged_df['error_y_abs'], 'Absolute Y-axis Error', 'green', 's--', average_error_y),
        (merged_df['error_z_abs'], 'Absolute Z-axis Error', 'blue', '^--', average_error_z)
    ]
    
    axs_flat = axs.flatten()
    for i, (error_series, title, color, marker, avg_error) in enumerate(error_data):
        ax = axs_flat[i]
        # 绘制主线
        ax.plot(marker_ids, error_series, marker, label=title, color=color, linewidth=2, markersize=8)
        
        # 绘制平均误差线
        ax.axhline(avg_error, color='gray', linestyle=':', linewidth=1.5, label=f'Mean Error ({avg_error:.4f})')
        
        ax.set_title(f'{title} (Mean: {avg_error:.4f} mm)', fontsize=18, weight='bold') # 放大到 18
        ax.set_ylabel('Absolute Error (mm)', fontsize=14) # 放大到 14
        ax.set_ylim(bottom=0)
        ax.grid(False) # <--- **修改：移除 2x2 误差图的网格线**
        ax.legend(fontsize=12) # 放大到 12
        ax.tick_params(labelsize=12) # 放大到 12
        
        if i >= 2: # 仅为底行添加X轴标签
            ax.set_xlabel('Marker ID', fontsize=14) # 放大到 14
            ax.set_xticks(marker_ids)
            ax.set_xticklabels(marker_ids, rotation=45, ha='right')

    # --- 6. 保存两张分析图 ---
    print("\n--- 步骤 6/6: 保存两张分析图 ---")
    path_3d = os.path.join(base_dir, 'sci_pose_comparison_3d.png') # 更改文件名以区分
    path_2x2_error = os.path.join(base_dir, 'sci_error_analysis.png')
    try:
        # 使用 bbox_inches='tight' 和更高的 DPI
        fig_3d.savefig(path_3d, dpi=400, bbox_inches='tight')
        print(f"-> 优化后的 3D 对比图已保存到: {path_3d}")
        fig_error.savefig(path_2x2_error, dpi=400, bbox_inches='tight')
        print(f"-> 优化后的 2x2 误差图已保存到: {path_2x2_error}")
    except Exception as e:
        print(f"--> 错误: 保存图像失败: {e}")

    # --- 7. 显示所有图像 ---
    print("\n--- 步骤 7/7: 显示所有生成的图像 ---")
    print("-> 正在显示图像... 关闭所有图像窗口即可退出程序。")
    plt.show()

if __name__ == "__main__":
    visualize_and_compare_with_alignment()
    print("\n程序执行完毕。")
