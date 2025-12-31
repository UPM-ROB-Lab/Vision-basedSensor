import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os
import imageio  # 用于创建GIF (Used for creating GIFs)
import chardet  # 用于检测文件编码 (Used for detecting file encoding)
import traceback # 用于打印详细的错误信息 (Used for printing detailed error info)

# ------------------------------
# 全局配置参数 (Global Configuration Parameters)
# ------------------------------
# REAL_MARKER_DIAMETER_MM = 3.0 # 5*5阵列marker直径
REAL_MARKER_DIAMETER_MM = 2.0 # circle阵列marker直径

# --- 输入数据的配置 (Configuration for input data) ---
# 重要：请根据您CSV文件中的实际列名（去除前后空格后），调整此字典的“键”。
# 脚本会将这些列重命名为内部使用的名称（'u', 'v', 'major_axis'）。
COLUMN_MAPPING = {
    'Cx': 'u',            # CSV中X坐标的列名 (Column name for X-coordinate in CSV)
    'Cy': 'v',            # CSV中Y坐标的列名 (Column name for Y-coordinate in CSV)
    'major_axis': 'major_axis' # CSV中标记点长轴的列名 (Column name for marker's major axis in CSV)
}

# =============================================================================
# 核心辅助函数 (Core Helper Functions)
# =============================================================================

def load_intrinsics_from_excel(filepath):
    """从指定的Excel文件加载相机内参矩阵和畸变系数。"""
    try:
        df = pd.read_excel(filepath)
        params = df.set_index('Parameter')['Value']
        camera_matrix = np.array([
            [params['fx'], params.get('skew', 0), params['cx']],
            [0,            params['fy'],          params['cy']],
            [0,            0,                     1]
        ], dtype=np.float32)
        dist_keys = ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6']
        dist_coeffs_list = [params[key] for key in dist_keys if key in params.index]
        dist_coeffs = np.array(dist_coeffs_list, dtype=np.float32)
        return camera_matrix, dist_coeffs
    except Exception as e:
        print(f"--> ERROR: Failed to load intrinsics file '{filepath}': {e}")
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
        print(f"--> ERROR: Failed to load extrinsics file '{filepath}': {e}")
        return None, None

def load_and_prepare_marker_data(filepath, column_mapping):
    """
    鲁棒地加载、解析、清理和准备标记点数据。
    - 自动检测文件编码
    - 使用灵活的分隔符
    - 清理列名中的空格
    - 验证并重命名列
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    with open(filepath, 'rb') as f:
        raw_data = f.read(30000)
    encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
    print(f"-> Detected file encoding: {encoding}")

    try:
        df = pd.read_csv(filepath, sep=r'\s+|,|\t', encoding=encoding, engine='python', skipinitialspace=True)
    except Exception as e:
        raise ValueError(f"Error parsing CSV file '{filepath}': {e}")

    df.columns = [str(col).strip() for col in df.columns]

    required_source_cols = set(column_mapping.keys()) | {'frameno', 'row', 'col'}
    if not required_source_cols.issubset(df.columns):
        missing = required_source_cols - set(df.columns)
        raise ValueError(f"CSV file is missing required columns: {missing}. Available columns: {list(df.columns)}")

    df.rename(columns=column_mapping, inplace=True)
    print("-> Marker data loaded and prepared successfully.")
    return df

# =============================================================================
# 动态位移计算的核心函数 (Core Function for Dynamic Displacement Calculation)
# =============================================================================
def calculate_displacement(initial_state, current_state, camera_matrix, R_wc, T_wc):
    """
    计算单个marker从初始状态到当前状态的三维位移。
    """
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    f_avg = (fx + fy) / 2.0
    u1, v1, x1 = initial_state['u'], initial_state['v'], initial_state['major_axis']
    u2, v2, x2 = current_state['u'], current_state['v'], current_state['major_axis']
    R1 = np.sqrt((u1 - cx)**2 + (v1 - cy)**2)
    # d1_effective = (REAL_MARKER_DIAMETER_MM / np.sqrt(R1**2 + f_avg**2)) * f_avg
    d1_effective = (REAL_MARKER_DIAMETER_MM / f_avg) * np.sqrt(R1**2 + f_avg**2)
    if x1 < 1e-6: return None, None, None
    h1 = f_avg * (d1_effective / x1)
    R2 = np.sqrt((u2 - cx)**2 + (v2 - cy)**2)
    if x1 < 1e-6: return None, None, None
    alpha = (x2 - x1) / x1
    term_ratio = np.sqrt((R1**2 + f_avg**2) / (R2**2 + f_avg**2))
    if abs(alpha + 1) < 1e-6: return None, None, None
    delta_h = h1 * (1 - (term_ratio / (alpha + 1)))
    Xc1 = h1 * (u1 - cx) / fx
    Yc1 = h1 * (v1 - cy) / fy
    P_cam1 = np.array([Xc1, Yc1, h1]).reshape(3, 1)
    P_world1 = (R_wc.T @ (P_cam1 - T_wc)).flatten()
    h2 = h1 - delta_h
    Xc2 = h2 * (u2 - cx) / fx
    Yc2 = h2 * (v2 - cy) / fy
    P_cam2 = np.array([Xc2, Yc2, h2]).reshape(3, 1)
    P_world2 = (R_wc.T @ (P_cam2 - T_wc)).flatten()
    displacement_vector = P_world2 - P_world1
    return P_world2, P_world1, displacement_vector

# =============================================================================
# 主程序 - 动态测量与可视化 (Main Program - Dynamic Measurement and Visualization)
# =============================================================================
def run_dynamic_measurement_and_visualization():
    """
    执行完整的动态测量流程，保存结果到Excel和GIF。
    """
    # --- 0. 定义文件路径 (Define file paths) ---
    data_base_dir = os.path.join("Results", "data")
    param_base_dir = os.path.join(data_base_dir, "PreprocessPara")
    output_base_dir = os.path.join(data_base_dir, "results")
    output_plot_dir = os.path.join(output_base_dir, "Displacement_Analysis_Plots")
    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(output_plot_dir, exist_ok=True)

    raw_marker_csv = os.path.join(data_base_dir, 'marker_locations_0.csv')
    intrinsic_params_xlsx = os.path.join(param_base_dir, "IntrinsicParameters.xlsx")
    extrinsic_params_xlsx = os.path.join(param_base_dir, "ExtrinsicParameters.xlsx")
    
    output_excel_path = os.path.join(output_base_dir, 'marker_3d_coordinates.xlsx')
    output_gif_path = os.path.join(output_base_dir, 'marker_displacement.gif')

    try:
        # --- 1. 加载所有参数和数据 (Load all parameters and data) ---
        print("--- Step 1: Loading Data and Static Parameters ---")
        camera_matrix, dist_coeffs = load_intrinsics_from_excel(intrinsic_params_xlsx)
        R_wc, T_wc = load_extrinsics_from_excel(extrinsic_params_xlsx)
        
        all_frames_df = load_and_prepare_marker_data(raw_marker_csv, COLUMN_MAPPING)

        if camera_matrix is None or R_wc is None:
            raise ValueError("Camera parameter loading failed.")

        print("-> Undistorting all 2D coordinates...")
        pixel_points = all_frames_df[['u', 'v']].values.astype(np.float32).reshape(-1, 1, 2)
        undistorted_points = cv2.undistortPoints(pixel_points, camera_matrix, dist_coeffs, None, camera_matrix)
        all_frames_df[['u', 'v']] = undistorted_points.reshape(-1, 2)
        print("-> Undistortion complete.")

        # --- 2. 提取初始状态 (第一帧) - Extract initial state (first frame) ---
        print("\n--- Step 2: Recording Initial State (First Frame) ---")
        first_frame_num = all_frames_df['frameno'].min()
        initial_frame_df = all_frames_df[all_frames_df['frameno'] == first_frame_num].copy()
        if initial_frame_df.empty:
            raise ValueError(f"Could not find the first frame (frameno={first_frame_num}) to use as a reference.")
        
        initial_frame_df.sort_values(by=['row', 'col'], inplace=True)
        initial_frame_df['marker_id'] = range(1,len(initial_frame_df)+1)
        initial_states = {row['marker_id']: row.to_dict() for _, row in initial_frame_df.iterrows()}
        print(f"-> Successfully recorded the initial state for {len(initial_states)} markers.")

        # --- 3. 逐帧计算位移并保存结果 (Process each frame to calculate displacement) ---
        print("\n--- Step 3: Calculating Displacement Frame-by-Frame ---")
        all_results = []
        frame_numbers = sorted(all_frames_df['frameno'].unique())
        
        for frame in frame_numbers:
            print(f"  - Processing frame {int(frame)}...")
            current_frame_df = all_frames_df[all_frames_df['frameno'] == frame].copy()
            current_frame_df.sort_values(by=['row', 'col'], inplace=True)
            current_frame_df['marker_id'] = range(1,len(current_frame_df)+1)

            for _, current_row in current_frame_df.iterrows():
                marker_id = current_row['marker_id']
                if marker_id in initial_states:
                    p_world2, _, displacement = calculate_displacement(
                        initial_states[marker_id], current_row.to_dict(), camera_matrix, R_wc, T_wc
                    )
                    if p_world2 is not None:
                        all_results.append({
                            'frameno': frame, 'marker_id': marker_id,
                            'Xw': p_world2[0], 'Yw': p_world2[1], 'Zw': p_world2[2],
                            'dX': displacement[0], 'dY': displacement[1], 'dZ': displacement[2]
                        })

        results_df = pd.DataFrame(all_results)
        results_df.to_excel(output_excel_path, index=False, float_format='%.6f')
        print(f"\n-> 3D coordinates for all frames successfully saved to: '{output_excel_path}'")

        # --- 4. 【修正】计算并绘制每个标记点的帧间位移 ---
        # --- [CORRECTED] Calculate and plot frame-to-frame displacement for each marker ---
        print("\n--- Step 4: Analyzing and Plotting Frame-to-Frame Displacement for Each Marker ---")
        
        # 按 marker_id 分组，然后计算每个marker在相邻帧之间的坐标差值
        # Group by marker_id, then calculate the coordinate difference for each marker between consecutive frames
        # 首先确保数据按 marker 和 帧号 正确排序
        # First, ensure data is sorted correctly by marker and then by frame
        plot_df = results_df.sort_values(by=['marker_id', 'frameno']).copy()
        
        # 对每个marker_id组进行差分操作
        # Perform the diff operation on each marker_id group
        diffs = plot_df.groupby('marker_id')[['Xw', 'Yw', 'Zw']].diff()
        diffs.rename(columns={'Xw': 'dX_frame', 'Yw': 'dY_frame', 'Zw': 'dZ_frame'}, inplace=True)
        
        # 将差值合并回DataFrame
        # Merge the differences back into the DataFrame
        plot_df = pd.concat([plot_df.reset_index(drop=True), diffs.reset_index(drop=True)], axis=1)
        
        # 计算三维欧氏距离（帧间位移大小）
        # Calculate the 3D Euclidean distance (frame-to-frame displacement magnitude)
        plot_df['inter_frame_displacement'] = np.sqrt(
            plot_df['dX_frame']**2 + plot_df['dY_frame']**2 + plot_df['dZ_frame']**2
        )
        
        marker_ids = plot_df['marker_id'].unique()
        plots_generated = 0
        # 为每个marker生成一张位移图
        # Generate one displacement plot for each marker
        for marker_id in marker_ids:
            marker_data = plot_df[plot_df['marker_id'] == marker_id].copy()
            
            # 检查是否有有效数据用于绘图 (第一帧的位移是NaN，会被忽略)
            # Check if there is valid data to plot (displacement for the first frame is NaN and will be ignored)
            if marker_data['inter_frame_displacement'].notna().any():
                fig, ax = plt.subplots(figsize=(12, 6))
                # 从第二帧开始绘制，因为第一帧没有“上一帧”
                # Plot from the second frame onwards, as the first frame has no "previous" frame
                ax.plot(marker_data['frameno'], marker_data['inter_frame_displacement'], marker='o', linestyle='-', markersize=4)
                
                ax.set_title(f'Frame-to-Frame Displacement of Marker {marker_id}')
                ax.set_xlabel('Frame Number')
                ax.set_ylabel('3D Displacement from Previous Frame (mm)')
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                # 设置Y轴从0开始，使视觉效果更直观
                # Set Y-axis to start from 0 for better visual interpretation
                ax.set_ylim(bottom=0)
                plt.tight_layout()
                
                # 定义并保存图像文件
                # Define and save the plot file
                plot_save_path = os.path.join(output_plot_dir, f'marker_{marker_id}_frame_displacement.png')
                plt.savefig(plot_save_path, dpi=150)
                plt.close(fig)
                plots_generated += 1

        print(f"-> Successfully generated and saved {plots_generated} frame-to-frame displacement plots to: '{output_plot_dir}'")


    #     # --- 5. 创建并保存GIF动画 (Create and save GIF animation) ---
    #     print("\n--- Step 5: Creating and Saving GIF Animation ---")
    #     gif_images = []
        
    #     initial_world_points_dict = {
    #         mid: calculate_displacement(s, s, camera_matrix, R_wc, T_wc)[1] 
    #         for mid, s in initial_states.items()
    #         if calculate_displacement(s, s, camera_matrix, R_wc, T_wc)[1] is not None
    #     }
    #     initial_world_points = np.array(list(initial_world_points_dict.values()))

    #     all_points = results_df[['Xw', 'Yw', 'Zw']].values
    #     if all_points.size == 0:
    #         raise ValueError("No 3D points were calculated, cannot create GIF.")
    #     plot_min = all_points.min(axis=0) - 5
    #     plot_max = all_points.max(axis=0) + 5
        
    #     for frame in frame_numbers:
    #         fig = plt.figure(figsize=(10, 8))
    #         ax = fig.add_subplot(111, projection='3d')
            
    #         ax.scatter(initial_world_points[:, 0], initial_world_points[:, 1], initial_world_points[:, 2],
    #                    c='gray', marker='o', s=30, alpha=0.5, label='Initial Position')
                       
    #         frame_data = results_df[results_df['frameno'] == frame]
    #         ax.scatter(frame_data['Xw'], frame_data['Yw'], frame_data['Zw'],
    #                    c='blue', marker='o', s=60, edgecolors='k', label=f'Current Position (Frame {int(frame)})')
                       
    #         for _, row in frame_data.iterrows():
    #             marker_id = row['marker_id']
    #             if marker_id in initial_world_points_dict:
    #                 initial_pos = initial_world_points_dict[marker_id]
    #                 current_pos = row[['Xw', 'Yw', 'Zw']].values
    #                 ax.plot([initial_pos[0], current_pos[0]], [initial_pos[1], current_pos[1]], [initial_pos[2], current_pos[2]], 'r-', alpha=0.6)

    #         ax.set_xlim(plot_min[0], plot_max[0]); ax.set_ylim(plot_min[1], plot_max[1]); ax.set_zlim(plot_min[2], plot_max[2])
    #         ax.set_xlabel('World X (mm)'); ax.set_ylabel('World Y (mm)'); ax.set_zlabel('World Z (mm)')
    #         ax.set_title(f'Marker Positions at Frame {int(frame)}')
    #         ax.legend(); ax.set_aspect('equal', 'box')
            
    #         fig.canvas.draw()
    #         image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #         gif_images.append(image)
    #         plt.close(fig)

    #     imageio.mimsave(output_gif_path, gif_images, fps=10) # 10 fps is a good default
    #     print(f"-> Dynamic displacement GIF successfully saved to: '{output_gif_path}'")

    except (FileNotFoundError, ValueError, Exception) as e:
        # 打印更详细的错误追踪信息 (Print more detailed traceback info)
        print(f"\n[FATAL ERROR] An error occurred during execution: {e}")
        print("--- Traceback ---")
        traceback.print_exc()
        print("-----------------")


if __name__ == "__main__":
    run_dynamic_measurement_and_visualization()
    print("\nProgram execution finished.")