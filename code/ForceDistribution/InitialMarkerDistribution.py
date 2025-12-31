import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import traceback

# ------------------------------
# 可视化配置参数 (Visualization Configuration)
# ------------------------------
TARGET_FRAME = 0 # 目标帧编号 (用于第一个函数)

# --- 新增位移图配置 ---
TARGET_MARKER_ID = 9 # 目标标记点编号 (例如：1)
# 模式选择: 'XYZ' (绘制X, Y, Z轴随时间变化) 或 'SCALAR' (绘制相对于起始点的总位移)
DISPLACEMENT_MODE = 'SCALAR' 

# =============================================================================
# 核心辅助函数 (SCI 风格绘图所需)
# =============================================================================

def set_axes_equal(ax):
    """确保3D绘图的X, Y, Z轴具有相同的尺度，防止几何失真。"""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

# =============================================================================
# 主程序 1 - 绘制第 0 帧坐标及编号 (原函数)
# =============================================================================
def plot_frame_zero_coordinates():
    """
    读取 marker_3d_coordinates.xlsx，绘制 TARGET_FRAME (第 0 帧) 的所有标记点坐标，
    并在每个点旁边标注其 marker_id。
    """
    # --- 0. 定义文件路径 (Define file paths) ---
    base_dir = os.path.join("data")
    input_excel_path = os.path.join(base_dir, 'marker_3d_coordinates.xlsx')
    output_plot_path = os.path.join(base_dir, f'SCI_Frame_{TARGET_FRAME}_Coordinates_Labeled.png')

    os.makedirs(base_dir, exist_ok=True)

    try:
        # --- 1. 加载数据 ---
        print(f"--- Step 1: Loading data and filtering for Frame {TARGET_FRAME} ---")
        if not os.path.exists(input_excel_path):
            raise FileNotFoundError(f"Input file not found: '{input_excel_path}'.")
        
        df = pd.read_excel(input_excel_path)
        
        # 提取目标帧的数据
        frame_data = df[df['frameno'] == TARGET_FRAME]

        if frame_data.empty:
            print(f"-> WARNING: Could not find data for Frame {TARGET_FRAME}. Aborting Frame 0 plot.")
            return
            
        print(f"-> Found {len(frame_data)} markers in Frame {TARGET_FRAME}.")

        # --- 2. 绘制 SCI 风格的三维坐标图并标注编号 ---
        print("--- Step 2: Creating and saving the labeled 3D coordinate plot ---")
        
        plt.style.use('seaborn-v0_8-whitegrid') 
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 1. 绘制标记点
        ax.scatter(
            frame_data['Xw'], frame_data['Yw'], frame_data['Zw'], 
            c='k', marker='o', s=50, alpha=0.8, 
            label=f'Marker Position (Frame {TARGET_FRAME})'
        )
        
        # 2. 标注每个点的编号
        for index, row in frame_data.iterrows():
            x, y, z = row['Xw'], row['Yw'], row['Zw']
            marker_id = row['marker_id']
            
            ax.text(
                x + 0.5, y + 0.5, z, 
                str(marker_id), 
                color='red', 
                fontsize=10, 
                weight='bold'
            )
        
        # 3. 设置专业化标签、标题和等比例尺
        ax.set_xlabel('World X (mm)', fontsize=14); 
        ax.set_ylabel('World Y (mm)', fontsize=14); 
        ax.set_zlabel('World Z (mm)', fontsize=14)
        ax.set_title(f'3D Marker Coordinates in Frame {TARGET_FRAME} (Labeled)', fontsize=16, weight='bold')
        ax.legend(loc='best', fontsize=12)
        ax.tick_params(labelsize=12)
        
        set_axes_equal(ax) # 应用等比例尺
        fig.tight_layout()

        # 使用高 DPI 保存
        plt.savefig(output_plot_path, dpi=400, bbox_inches='tight') 
        print(f"-> Labeled 3D plot saved to: '{output_plot_path}'")

        # --- 3. 显示绘图 ---
        # plt.show() # 暂时注释，避免与第二个函数的plt.show()冲突，统一在最后显示

    except (FileNotFoundError, Exception) as e:
        print(f"\n[FATAL ERROR] An error occurred during Frame 0 visualization: {e}")
        print("--- Traceback ---")
        traceback.print_exc()
        print("-----------------")


# =============================================================================
# 主程序 2 - 绘制指定标记点的位移图 (新增功能)
# =============================================================================

def plot_marker_displacement(marker_id, mode):
    """
    绘制指定 marker_id 在所有帧中的位移图。
    mode: 'XYZ' 或 'SCALAR'。
    """
    base_dir = os.path.join("data")
    input_excel_path = os.path.join(base_dir, 'marker_3d_coordinates.xlsx')

    try:
        print(f"\n--- Step 4: Plotting Displacement for Marker {marker_id} (Mode: {mode}) ---")
        if not os.path.exists(input_excel_path):
            raise FileNotFoundError(f"Input file not found: '{input_excel_path}'.")
        
        df = pd.read_excel(input_excel_path)
        
        # 过滤出目标标记点的数据
        marker_data = df[df['marker_id'] == marker_id].sort_values(by='frameno').reset_index(drop=True)

        if marker_data.empty:
            print(f"-> WARNING: Could not find data for Marker ID {marker_id}. Aborting displacement plot.")
            return
            
        print(f"-> Found {len(marker_data)} frames for Marker {marker_id}.")

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        frames = marker_data['frameno']
        
        if mode == 'XYZ':
            # 绘制 X, Y, Z 坐标随时间的变化
            ax.plot(frames, marker_data['Xw'], label='X Position (mm)', linewidth=2)
            ax.plot(frames, marker_data['Yw'], label='Y Position (mm)', linewidth=2)
            ax.plot(frames, marker_data['Zw'], label='Z Position (mm)', linewidth=2)
            
            ax.set_ylabel('Position (mm)', fontsize=14)
            title = f'Position of Marker {marker_id} Over Time (X, Y, Z)'
            
        elif mode == 'SCALAR':
            # 计算相对于起始点 (Frame 0) 的标量位移
            
            # 1. 获取起始点坐标
            start_pos = marker_data[marker_data['frameno'] == 0]
            if start_pos.empty:
                print("-> ERROR: Frame 0 data missing for displacement calculation.")
                return
            
            X0, Y0, Z0 = start_pos[['Xw', 'Yw', 'Zw']].iloc[0].values
            
            # 2. 计算每帧的位移 (欧氏距离)
            displacement = np.sqrt(
                (marker_data['Xw'] - X0)**2 + 
                (marker_data['Yw'] - Y0)**2 + 
                (marker_data['Zw'] - Z0)**2
            )
            
            # 3. 绘图
            ax.plot(frames, displacement, label='Total Displacement from Frame 0 (mm)', color='purple', linewidth=3)
            
            ax.set_ylabel('Displacement Magnitude (mm)', fontsize=14)
            title = f'Scalar Displacement of Marker {marker_id} from Start Point'
            
        else:
            print(f"-> ERROR: Invalid DISPLACEMENT_MODE '{mode}'. Use 'XYZ' or 'SCALAR'.")
            return

        # 设置通用图表属性
        ax.set_xlabel('Frame Number', fontsize=14)
        ax.set_title(title, fontsize=16, weight='bold')
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()
        
        print("--- Step 5: Displaying displacement plot on screen ---")
        plt.show()

    except (FileNotFoundError, Exception) as e:
        print(f"\n[FATAL ERROR] An error occurred during displacement visualization: {e}")
        print("--- Traceback ---")
        traceback.print_exc()
        print("-----------------")


if __name__ == "__main__":
    # 运行第一个函数：绘制第 0 帧的坐标和编号
    plot_frame_zero_coordinates()
    
    # 运行第二个函数：绘制指定标记点的位移图
    plot_marker_displacement(TARGET_MARKER_ID, DISPLACEMENT_MODE)
    
    print("\nAll visualization tasks finished.")