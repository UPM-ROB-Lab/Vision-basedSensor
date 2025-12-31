import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import traceback

# ------------------------------
# 分析配置参数 (Analysis Configuration)
# ------------------------------
# 感兴趣的标记点编号列表
TARGET_MARKERS = [8,9,10,11,12,13,14,15,16,17,18,19] 
# 初始帧范围 (包含起始和结束)
START_FRAME_RANGE = (1, 30) 
# 结束帧范围 (包含起始和结束)
END_FRAME_RANGE = (120, 150) 

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
# 主程序 - 基于平均坐标的位移分析
# =============================================================================
def plot_averaged_displacement():
    """
    计算指定标记点在两个帧范围内的平均坐标，并绘制平均位移矢量图。
    """
    # --- 0. 定义文件路径 (Define file paths) ---
    base_dir = os.path.join("data")
    input_excel_path = os.path.join(base_dir, 'marker_3d_coordinates.xlsx')
    
    # 输出文件名反映了分析的性质
    output_plot_path = os.path.join(
        base_dir, 
        f'Averaged_Displacement.png'
    )

    os.makedirs(base_dir, exist_ok=True)

    try:
        # --- 1. 加载数据并筛选标记点 ---
        print("--- Step 1: Loading data and filtering markers ---")
        if not os.path.exists(input_excel_path):
            raise FileNotFoundError(f"Input file not found: '{input_excel_path}'.")
        
        df = pd.read_excel(input_excel_path)
        
        # 筛选出感兴趣的标记点
        df_filtered = df[df['marker_id'].isin(TARGET_MARKERS)]
        
        if df_filtered.empty:
            print("-> WARNING: No data found for the target markers. Aborting.")
            return

        # --- 2. 计算初始平均坐标 (Start Average) ---
        print(f"--- Step 2: Calculating average coordinates for Start Range {START_FRAME_RANGE} ---")
        
        start_data = df_filtered[
            (df_filtered['frameno'] >= START_FRAME_RANGE[0]) & 
            (df_filtered['frameno'] <= START_FRAME_RANGE[1])
        ]
        
        # 按 marker_id 分组并计算平均值
        start_avg = start_data.groupby('marker_id')[['Xw', 'Yw', 'Zw']].mean().reset_index()
        start_avg.columns = ['marker_id', 'X_start', 'Y_start', 'Z_start']


        # --- 3. 计算结束平均坐标 (End Average) ---
        print(f"--- Step 3: Calculating average coordinates for End Range {END_FRAME_RANGE} ---")
        
        end_data = df_filtered[
            (df_filtered['frameno'] >= END_FRAME_RANGE[0]) & 
            (df_filtered['frameno'] <= END_FRAME_RANGE[1])
        ]
        
        # 按 marker_id 分组并计算平均值
        end_avg = end_data.groupby('marker_id')[['Xw', 'Yw', 'Zw']].mean().reset_index()
        end_avg.columns = ['marker_id', 'X_end', 'Y_end', 'Z_end']

        # --- 4. 合并数据并计算矢量 ---
        
        # 合并起始和结束的平均坐标
        merged_data = pd.merge(start_avg, end_avg, on='marker_id', how='inner')
        
        if merged_data.empty:
            print("-> WARNING: No common markers found with data in both ranges. Aborting.")
            return

        # 计算位移矢量 (Delta X, Delta Y, Delta Z)
        merged_data['dX'] = merged_data['X_end'] - merged_data['X_start']
        merged_data['dY'] = merged_data['Y_end'] - merged_data['Y_start']
        merged_data['dZ'] = merged_data['Z_end'] - merged_data['Z_start']
        
        # 计算平均位移大小
        displacement_magnitudes = np.linalg.norm(merged_data[['dX', 'dY', 'dZ']].values, axis=1)
        mean_displacement = np.mean(displacement_magnitudes)
        
        print(f"-> Successfully calculated vectors for {len(merged_data)} markers. Mean displacement: {mean_displacement:.4f} mm.")


        # --- 5. 绘制 SCI 风格的三维位移矢量图 ---
        print("--- Step 4: Creating and saving the SCI-style 3D displacement plot ---")
        
        plt.style.use('seaborn-v0_8-whitegrid') 
        fig_3d = plt.figure(figsize=(10, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        
        # 1. 绘制起点 (平均初始位置)
        ax_3d.scatter(
            merged_data['X_start'], merged_data['Y_start'], merged_data['Z_start'], 
            c='blue', marker='o', s=80, edgecolors='k', linewidths=0.5, alpha=0.6, 
            label=f'Avg Start Position (Frames {START_FRAME_RANGE[0]}-{START_FRAME_RANGE[1]})'
        )
        
        # 2. 绘制终点 (平均结束位置)
        ax_3d.scatter(
            merged_data['X_end'], merged_data['Y_end'], merged_data['Z_end'], 
            c='red', marker='P', s=100, alpha=0.8, 
            label=f'Avg End Position (Frames {END_FRAME_RANGE[0]}-{END_FRAME_RANGE[1]})'
        )
        
        # 3. 绘制位移矢量箭头
        ax_3d.quiver(
            merged_data['X_start'], merged_data['Y_start'], merged_data['Z_start'],
            merged_data['dX'], merged_data['dY'], merged_data['dZ'],
            color='green', arrow_length_ratio=0.1, linewidth=2.0, 
            label='Averaged Displacement Vector', alpha=0.8
        )
        
        # 4. 标注标记点编号 (可选，但有助于识别哪个箭头对应哪个点)
        for index, row in merged_data.iterrows():
            ax_3d.text(
                row['X_start'], row['Y_start'], row['Z_start'] + 1, # 稍微抬高 Z 轴
                f"M{int(row['marker_id'])}", 
                color='purple', fontsize=9, weight='bold'
            )

        # 5. 设置专业化标签、标题和等比例尺
        ax_3d.set_xlabel('World X (mm)', fontsize=14); 
        ax_3d.set_ylabel('World Y (mm)', fontsize=14); 
        ax_3d.set_zlabel('World Z (mm)', fontsize=14)
        ax_3d.set_title(
            f'Averaged 3D Displacement for Markers {TARGET_MARKERS}\n'
            f'(Start Avg: F{START_FRAME_RANGE[0]}-{START_FRAME_RANGE[1]} to End Avg: F{END_FRAME_RANGE[0]}-{END_FRAME_RANGE[1]})', 
            fontsize=14, weight='bold'
        )
        ax_3d.legend(loc='best', fontsize=10)
        ax_3d.tick_params(labelsize=12)
        
        set_axes_equal(ax_3d) # 应用等比例尺
        fig_3d.tight_layout()

        # 使用高 DPI 保存
        plt.savefig(output_plot_path, dpi=400, bbox_inches='tight') 
        print(f"-> SCI-style Averaged 3D plot saved to: '{output_plot_path}'")

        # --- 6. 显示绘图 ---
        print("\n--- Step 5: Displaying plot on screen ---")
        plt.show()

    except (FileNotFoundError, Exception) as e:
        print(f"\n[FATAL ERROR] An error occurred during averaged displacement visualization: {e}")
        print("--- Traceback ---")
        traceback.print_exc()
        print("-----------------")


if __name__ == "__main__":
    plot_averaged_displacement()
    print("\nAveraged displacement analysis finished.")