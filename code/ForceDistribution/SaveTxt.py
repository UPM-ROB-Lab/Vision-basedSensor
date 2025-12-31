import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 引入 3D 绘图模块
import os
import traceback

# ------------------------------
# 分析配置参数 (Analysis Configuration)
# ------------------------------
# 感兴趣的标记点编号列表 (示例数据)
TARGET_MARKERS = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] 
# 初始帧范围 (包含起始和结束)
START_FRAME_RANGE = (34, 100) 
# 结束帧范围 (包含起始和结束)
END_FRAME_RANGE = (3418,3569) 

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
# 主程序 - 计算、保存并显示位移矢量
# =============================================================================
def calculate_save_and_plot_averaged_displacement():
    """
    计算指定标记点在两个帧范围内的平均坐标，保存到 TXT 文件，并显示 3D 位移矢量图。
    """
    # --- 0. 定义文件路径 (Define file paths) ---
    base_dir = os.path.join("data")
    input_excel_path = os.path.join(base_dir, 'marker_3d_coordinates.xlsx')
    # output_txt_path = os.path.join(base_dir, 'initial_many.txt') # 初始姿态
    output_txt_path = os.path.join(base_dir, '7_many.txt') # 最终姿态

    os.makedirs(base_dir, exist_ok=True)

    try:
        # --- 1. 加载数据并筛选标记点 ---
        print("--- Step 1: Loading data and filtering markers ---")
        if not os.path.exists(input_excel_path):
            raise FileNotFoundError(f"Input file not found: '{input_excel_path}'.")
        
        df = pd.read_excel(input_excel_path)
        df_filtered = df[df['marker_id'].isin(TARGET_MARKERS)]
        
        if df_filtered.empty:
            print("-> WARNING: No data found for the target markers. Aborting.")
            return

        # --- 2. 计算初始平均坐标 (Start Average) ---
        start_data = df_filtered[
            (df_filtered['frameno'] >= START_FRAME_RANGE[0]) & 
            (df_filtered['frameno'] <= START_FRAME_RANGE[1])
        ]
        start_avg = start_data.groupby('marker_id')[['Xw', 'Yw', 'Zw']].mean().reset_index()
        start_avg.columns = ['marker_id', 'X_start', 'Y_start', 'Z_start']

        # --- 3. 计算结束平均坐标 (End Average) ---
        end_data = df_filtered[
            (df_filtered['frameno'] >= END_FRAME_RANGE[0]) & 
            (df_filtered['frameno'] <= END_FRAME_RANGE[1])
        ]
        end_avg = end_data.groupby('marker_id')[['Xw', 'Yw', 'Zw']].mean().reset_index()
        end_avg.columns = ['marker_id', 'X_end', 'Y_end', 'Z_end']

        # --- 4. 合并数据并计算矢量 ---
        merged_data = pd.merge(start_avg, end_avg, on='marker_id', how='inner')
        
        if merged_data.empty:
            print("-> WARNING: No common markers found with data in both ranges. Aborting.")
            return
            
        # 计算位移矢量 (Delta X, Delta Y, Delta Z)
        merged_data['dX'] = merged_data['X_end'] - merged_data['X_start']
        merged_data['dY'] = merged_data['Y_end'] - merged_data['Y_start']
        merged_data['dZ'] = merged_data['Z_end'] - merged_data['Z_start']
        
        displacement_magnitudes = np.linalg.norm(merged_data[['dX', 'dY', 'dZ']].values, axis=1)
        mean_displacement = np.mean(displacement_magnitudes)
        
        print(f"-> Successfully calculated vectors for {len(merged_data)} markers. Mean displacement: {mean_displacement:.4f} mm.")

        # --- 5. 保存平均坐标到 TXT 文件 ---
        print(f"--- Step 5: Saving averaged coordinates to '{output_txt_path}' ---")
        
        save_df = merged_data[['marker_id', 'X_start', 'Y_start', 'Z_start', 'X_end', 'Y_end', 'Z_end']]
        
        with open(output_txt_path, 'w') as f:
            f.write("Averaged Marker Coordinates (Units: mm)\n")
            f.write("--------------------------------------------------------------------------------\n")
            f.write("Start Avg Range: Frames {}-{} | End Avg Range: Frames {}-{}\n".format(
                START_FRAME_RANGE[0], START_FRAME_RANGE[1], END_FRAME_RANGE[0], END_FRAME_RANGE[1]
            ))
            f.write("Target Markers: {}\n".format(TARGET_MARKERS))
            f.write("--------------------------------------------------------------------------------\n")
            f.write(save_df.to_string(index=False, float_format="%.4f"))
            
        print("-> Data successfully saved.")

        # --- 6. 绘制 SCI 风格的三维位移矢量图 (Visualization) ---
        print("--- Step 6: Creating and displaying the 3D displacement plot for verification ---")
        
        plt.style.use('seaborn-v0_8-whitegrid') 
        fig_3d = plt.figure(figsize=(10, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        
        # 1. 绘制起点 (蓝色圆圈)
        ax_3d.scatter(
            merged_data['X_start'], merged_data['Y_start'], merged_data['Z_start'], 
            c='blue', marker='o', s=80, edgecolors='k', linewidths=0.5, alpha=0.6, 
            label=f'Avg Start Position (F{START_FRAME_RANGE[0]}-{START_FRAME_RANGE[1]})'
        )
        
        # 2. 绘制终点 (红色加号)
        ax_3d.scatter(
            merged_data['X_end'], merged_data['Y_end'], merged_data['Z_end'], 
            c='red', marker='+', s=100, alpha=0.8, 
            label=f'Avg End Position (F{END_FRAME_RANGE[0]}-{END_FRAME_RANGE[1]})'
        )
        
        # 3. 绘制位移矢量箭头 (绿色)
        ax_3d.quiver(
            merged_data['X_start'], merged_data['Y_start'], merged_data['Z_start'],
            merged_data['dX'], merged_data['dY'], merged_data['dZ'],
            color='green', arrow_length_ratio=0.1, linewidth=2.0, 
            label='Averaged Displacement Vector', alpha=0.8
        )
        
        # 4. 标注标记点编号
        for index, row in merged_data.iterrows():
            ax_3d.text(
                row['X_start'], row['Y_start'], row['Z_start'] + 0.5, 
                f"M{int(row['marker_id'])}", 
                color='purple', fontsize=9, weight='bold'
            )

        # 5. 设置专业化标签、标题和等比例尺
        ax_3d.set_xlabel('World X (mm)', fontsize=14); 
        ax_3d.set_ylabel('World Y (mm)', fontsize=14); 
        ax_3d.set_zlabel('World Z (mm)', fontsize=14)
        ax_3d.set_title(
            f'Averaged 3D Displacement Verification (Mean: {mean_displacement:.4f} mm)', 
            fontsize=14, weight='bold'
        )
        ax_3d.legend(loc='best', fontsize=10)
        ax_3d.tick_params(labelsize=12)
        
        set_axes_equal(ax_3d) # 应用等比例尺
        fig_3d.tight_layout()

        # --- 7. 显示绘图 ---
        print("\n--- Step 7: Displaying plot on screen ---")
        plt.show()

    except (FileNotFoundError, Exception) as e:
        print(f"\n[FATAL ERROR] An error occurred during analysis: {e}")
        print("--- Traceback ---")
        traceback.print_exc()
        print("-----------------")


if __name__ == "__main__":
    calculate_save_and_plot_averaged_displacement()
    print("\nAnalysis and visualization finished.")