import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import traceback
# from matplotlib.ticker import FormatStrFormatter # 暂时不需要，但保留以备后用

# ------------------------------
# 可视化配置参数 (Visualization Configuration)
# ------------------------------
START_FRAME = 1
END_FRAME = 120

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
# 主程序 - 最终可视化 (Main Program - Final Visualization)
# =============================================================================
def create_final_plots_sci_style():
    """
    读取后处理后的数据，创建、保存并显示一张SCI风格的三维和一张二维(XY)位移图。
    
    主要优化点：高 DPI, 等比例尺, 增大字体, 优化点标记和箭头样式。
    """
    # --- 0. 定义文件路径 (Define file paths) ---
    base_dir = os.path.join("Results", "data", "results")
    
    # 输入文件：后处理后的Excel表格
    input_excel_path = os.path.join(base_dir, 'marker_3d_coordinates.xlsx')

    # 输出文件路径 (使用 SCI 风格的固定文件名)
    output_3d_plot_path = os.path.join(base_dir, 'SCI_3D_Displacement.png')
    output_2d_plot_path = os.path.join(base_dir, 'SCI_2D_Displacement.png')

    os.makedirs(base_dir, exist_ok=True)

    try:
        # --- 1. 加载数据 ---
        print("--- Step 1: Loading post-processed 3D coordinate data ---")
        if not os.path.exists(input_excel_path):
            raise FileNotFoundError(f"Input file not found: '{input_excel_path}'. Please run the post-processing script first.")
        
        df = pd.read_excel(input_excel_path)
        print(f"-> Successfully loaded data from '{input_excel_path}'.")

        # --- 2. 提取数据和计算向量 ---
        print(f"--- Step 2: Extracting data and calculating vectors ---")
        
        start_frame_data = df[df['frameno'] == START_FRAME].set_index('marker_id')
        end_frame_data = df[df['frameno'] == END_FRAME].set_index('marker_id')

        if start_frame_data.empty or end_frame_data.empty:
            print(f"-> WARNING: Could not find data for both start frame {START_FRAME} and end frame {END_FRAME}. Aborting.")
            return

        common_markers = start_frame_data.index.intersection(end_frame_data.index)
        if common_markers.empty:
            print("-> WARNING: No common markers found between the two frames. Aborting.")
            return
            
        start_points = start_frame_data.loc[common_markers, ['Xw', 'Yw', 'Zw']]
        end_points = end_frame_data.loc[common_markers, ['Xw', 'Yw', 'Zw']]
        vectors = end_points.values - start_points.values
        
        # 计算位移的欧氏距离和平均值
        displacement_magnitudes = np.linalg.norm(vectors, axis=1)
        mean_displacement = np.mean(displacement_magnitudes)
        
        print(f"-> Found {len(common_markers)} common markers to plot. Mean displacement: {mean_displacement:.4f} mm.")


        # --- 3. 绘制 SCI 风格的三维位移矢量图 (3D Displacement Plot) ---
        print("--- Step 3: Creating and saving the SCI-style 3D displacement plot ---")
        
        # 应用 SCI 风格 (可选：'default', 'ggplot', 'seaborn-v0_8-whitegrid')
        plt.style.use('seaborn-v0_8-whitegrid') 
        fig_3d = plt.figure(figsize=(10, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        
        # 1. 绘制起点 (蓝色圆圈，带黑色边框，透明度适中)
        ax_3d.scatter(
            start_points['Xw'], start_points['Yw'], start_points['Zw'], 
            c='blue', marker='o', s=80, edgecolors='k', linewidths=0.5, alpha=0.6, 
            label=f'Start Position (Frame {START_FRAME})'
        )
        
        # 2. 绘制终点 (红色五角星，高对比度)
        ax_3d.scatter(
            end_points['Xw'], end_points['Yw'], end_points['Zw'], 
            c='red', marker='P', s=100, alpha=0.8, 
            label=f'End Position (Frame {END_FRAME})'
        )
        
        # 3. 绘制位移矢量箭头 (绿色，加粗)
        ax_3d.quiver(
            start_points['Xw'], start_points['Yw'], start_points['Zw'],
            vectors[:, 0], vectors[:, 1], vectors[:, 2],
            color='green', arrow_length_ratio=0.1, linewidth=2.0, 
            label='Displacement Vector', alpha=0.8
        )
        
        # 4. 设置专业化标签、标题和等比例尺
        ax_3d.set_xlabel('World X (mm)', fontsize=14); 
        ax_3d.set_ylabel('World Y (mm)', fontsize=14); 
        ax_3d.set_zlabel('World Z (mm)', fontsize=14)
        ax_3d.set_title(f'3D Displacement Vectors from Frame {START_FRAME} to {END_FRAME}\n(Mean Displacement: {mean_displacement:.4f} mm)', fontsize=16, weight='bold')
        ax_3d.legend(loc='best', fontsize=12)
        ax_3d.grid(False) # 移除 3D 图的网格线
        ax_3d.tick_params(labelsize=12)
        
        set_axes_equal(ax_3d) # 应用等比例尺
        fig_3d.tight_layout()

        # 使用高 DPI 保存
        plt.savefig(output_3d_plot_path, dpi=400, bbox_inches='tight') 
        print(f"-> SCI-style 3D plot saved to: '{output_3d_plot_path}'")


        # --- 4. 绘制 SCI 风格的二维(XY)投影图 (2D (XY) Projection Plot) ---
        print("--- Step 4: Creating and saving the SCI-style 2D (XY) displacement plot ---")
        
        fig_2d, ax_2d = plt.subplots(figsize=(10, 8))
        
        # 1. 绘制起点
        ax_2d.scatter(
            start_points['Xw'], start_points['Yw'], 
            c='blue', marker='o', s=80, edgecolors='k', linewidths=0.5, alpha=0.6, 
            label=f'Start Position (Frame {START_FRAME})'
        )
        
        # 2. 绘制终点
        ax_2d.scatter(
            end_points['Xw'], end_points['Yw'], 
            c='red', marker='P', s=100, alpha=0.8, 
            label=f'End Position (Frame {END_FRAME})'
        )
        
        # 3. 绘制二维箭头
        ax_2d.quiver(
            start_points['Xw'], start_points['Yw'],
            vectors[:, 0], # X component of vectors
            vectors[:, 1], # Y component of vectors
            color='green', angles='xy', scale_units='xy', scale=1, linewidth=2.0, 
            label='Displacement Vector', alpha=0.8
        )

        # 4. 设置专业化标签和等比例尺
        ax_2d.set_xlabel('X (mm)', fontsize=14)
        ax_2d.set_ylabel('Y (mm)', fontsize=14)
        ax_2d.set_title(f'2D Displacement (XY Plane) from Frame {START_FRAME} to {END_FRAME}', fontsize=16, weight='bold')
        ax_2d.legend(loc='best', fontsize=12)
        ax_2d.grid(True, linestyle='--', alpha=0.7) # 2D图保留优化后的网格线
        ax_2d.set_aspect('equal', 'box') # 保持等比例尺
        ax_2d.tick_params(labelsize=12)
        fig_2d.tight_layout()
        
        # 使用高 DPI 保存
        plt.savefig(output_2d_plot_path, dpi=400, bbox_inches='tight') 
        print(f"-> SCI-style 2D plot saved to: '{output_2d_plot_path}'")

        # --- 5. 显示所有绘图 ---
        print("\n--- Step 5: Displaying plots on screen ---")
        print("-> Close the plot windows to allow the program to finish.")
        plt.show()

    except (FileNotFoundError, Exception) as e:
        print(f"\n[FATAL ERROR] An error occurred during final visualization: {e}")
        print("--- Traceback ---")
        traceback.print_exc()
        print("-----------------")


if __name__ == "__main__":
    # 实际运行时请调用这个 SCI 风格的函数
    create_final_plots_sci_style()
    print("\nSCI-style visualization program execution finished.")