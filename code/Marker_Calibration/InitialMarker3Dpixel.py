import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse # 导入用于绘制椭圆的模块

def process_marker_data_and_save(txt_filepath, output_csv_path):
    """
    Reads raw marker data from a TXT file, calculates the average position (u, v),
    average major_axis, average minor_axis, and average angle for each marker, 
    and saves the results to a CSV file.
    """
    print(f"--- Stage 1: Processing Data ---")
    print(f"Reading raw data from: '{txt_filepath}'")

    # --- 1. 加载和解析原始TXT文件 ---
    try:
        # 【修改】增加 'minor_axis' (8) 和 'angle' (9) 列的读取
        col_indices_to_use = [0, 1, 2, 5, 6, 7, 8, 9] 
        col_names_for_selected = ['frameno', 'row', 'col', 'Cx', 'Cy', 'major_axis', 'minor_axis', 'angle']
        df = pd.read_csv(txt_filepath, header=None, skiprows=1, sep='\s+',
                         usecols=col_indices_to_use, names=col_names_for_selected, engine='python')
        if df.empty:
            print("Error: DataFrame is empty after reading the TXT file.")
            return False
    except FileNotFoundError:
        print(f"Error: Input file not found at '{txt_filepath}'.")
        return False
    except Exception as e:
        print(f"Error reading or parsing TXT file: {e}")
        return False

    # --- 2. 数据清洗 ---
    for col_name in col_names_for_selected:
        if df[col_name].dtype == 'object':
            df[col_name] = df[col_name].str.strip(',')
    
    for col_name in col_names_for_selected:
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    df.dropna(inplace=True)
    if df.empty:
        print("Error: DataFrame became empty after data cleaning and conversion.")
        return False

    # --- 3. 分组和计算平均值 ---
    print("Grouping by marker (row, col) and calculating averages...")
    grouped = df.groupby(['row', 'col'])
    
    # 计算平均形状参数和角度
    marker_avg_shape = grouped[['major_axis', 'minor_axis', 'angle']].mean().reset_index()
    marker_avg_shape.rename(columns={
        'major_axis': 'average_major_axis',
        'minor_axis': 'average_minor_axis',
        'angle': 'average_angle' # 新增
    }, inplace=True)
    
    # 计算平均像素坐标
    marker_avg_position = grouped[['Cx', 'Cy']].mean().reset_index()
    marker_avg_position.rename(columns={'Cx': 'avg_u', 'Cy': 'avg_v'}, inplace=True)
    
    # 合并结果
    processed_markers = pd.merge(marker_avg_shape, marker_avg_position, on=['row', 'col'])
    
    # --- 4. 分配ID并保存到CSV ---
    processed_markers.sort_values(by=['row', 'col'], inplace=True)
    processed_markers['marker_id'] = range(1, len(processed_markers) + 1)
    
    # 【修改】选择并重命名最终要保存的列，加入 minor_axis 和 angle
    columns_to_save = ['marker_id', 'avg_u', 'avg_v', 'average_major_axis', 'average_minor_axis', 'average_angle']
    final_df_to_save = processed_markers[columns_to_save].copy()
    final_df_to_save.rename(columns={
        'avg_u': 'u', 
        'avg_v': 'v',
        'average_major_axis': 'major_axis',
        'average_minor_axis': 'minor_axis',
        'average_angle': 'angle'
    }, inplace=True)
    
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_csv_path)
        os.makedirs(output_dir, exist_ok=True)
        
        final_df_to_save.to_csv(output_csv_path, index=False, float_format='%.4f')
        print(f"Successfully processed and saved averaged data to '{output_csv_path}'")
        return True
    except Exception as e:
        print(f"Error saving data to CSV: {e}")
        return False


def plot_pixel_coordinates_2d(pixel_csv_path):
    """
    Loads averaged pixel coordinates and shape parameters from the CSV 
    and plots them as ellipses in a 2D scatter plot.
    """
    print(f"\n--- Stage 2: Plotting Data ---")
    print(f"Reading averaged data from: '{pixel_csv_path}'")

    # --- 1. 加载处理后的数据 ---
    try:
        df = pd.read_csv(pixel_csv_path)
        # 检查所有必需的列，包括角度
        required_cols = ['marker_id', 'u', 'v', 'major_axis', 'minor_axis', 'angle']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: CSV '{pixel_csv_path}' is missing one of the required columns: {required_cols}.")
            return
    except FileNotFoundError:
        print(f"Error: Processed pixel file not found at '{pixel_csv_path}'.")
        return
    except Exception as e:
        print(f"Error loading pixel coordinates from CSV: {e}")
        return

    # --- 2. 创建绘图 ---
    fig, ax = plt.subplots(figsize=(12, 10))

    # 绘制散点图（作为背景或中心点）
    # 使用 minor_axis 作为颜色映射，以便区分形状
    sc = ax.scatter(df['u'], df['v'], c=df['minor_axis'], s=10, 
                    cmap='plasma', alpha=0.8, edgecolors='k', linewidth=0.5, zorder=2)

    cbar = plt.colorbar(sc)
    cbar.set_label('Average Minor Axis (pixels)')

    # --- 3. 绘制椭圆 ---
    for i, row in df.iterrows():
        # Matplotlib Ellipse 需要宽度和高度，以及角度（逆时针，以度为单位）
        # 注意：OpenCV 的 fitEllipse 返回的 width/height 是直径，Matplotlib 需要半径或直径，
        # 这里的 Ellipse 构造函数接受 width 和 height 作为直径。
        
        # Matplotlib 的角度定义：逆时针旋转，0度在正X轴。
        # OpenCV 的角度定义：通常是长轴与X轴的夹角，范围 [0, 180) 或 [0, 360)。
        # 我们使用 OpenCV 识别到的角度。
        
        # 绘制椭圆
        ellipse = Ellipse(
            xy=(row['u'], row['v']), 
            width=row['major_axis'], 
            height=row['minor_axis'], 
            angle=row['angle'], # 使用平均角度
            facecolor='none', 
            edgecolor='green', 
            linewidth=1.5,
            alpha=0.7,
            zorder=3 # 确保椭圆在散点图之上
        )
        ax.add_patch(ellipse)
        
        # 绘制长轴（可选，用于更清晰地显示方向）
        # 角度转换为弧度
        angle_rad = np.deg2rad(row['angle'])
        
        # 计算长轴端点
        half_major = row['major_axis'] / 2
        
        # 长轴方向向量
        dx = half_major * np.cos(angle_rad)
        dy = half_major * np.sin(angle_rad)
        
        # 绘制长轴线段 (使用红色)
        ax.plot([row['u'] - dx, row['u'] + dx], 
                [row['v'] - dy, row['v'] + dy], 
                color='red', 
                linewidth=1, 
                zorder=4)


    # --- 4. 设置坐标轴和标题 ---
    ax.set_xlabel('u-axis (pixels)')
    ax.set_ylabel('v-axis (pixels)')
    ax.set_title('2D Visualization of Averaged Marker Shapes and Positions')
    
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis() # 图像坐标系 V轴向下
    ax.grid(True, linestyle='--', alpha=0.6)

    # --- 5. 在每个点旁边标注其 marker_id ---
    for i, row in df.iterrows():
        # 稍微偏移标注，避免与椭圆重叠
        ax.text(row['u'] + row['major_axis'] * 0.6, row['v'], 
                f"{row['marker_id']}", fontsize=8, color='black', ha='left', va='center')

    # --- 6. 显示图像 ---
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # --- 定义文件路径 ---
    data_folder_path = os.path.join("Results", "data", 'PreprocessPara')
    os.makedirs(data_folder_path, exist_ok=True)
    
    # 输入的原始数据文件 (假设您的视频处理程序输出的CSV是这个格式)
    # 注意：如果您的视频处理程序输出的是 CSV，但您这里读取的是 TXT，请确保格式匹配。
    # 假设 'MarkerCalibration.txt' 实际上是视频处理程序输出的 CSV 文件，或者您已将其转换为 TXT 格式。
    txt_file_path = os.path.join(data_folder_path, 'MarkerCalibration.txt')
    
    # 处理后输出的CSV文件
    output_csv_file = os.path.join(data_folder_path, 'pixel_marker.csv')

    # --- 执行工作流 ---
    # 步骤1: 处理原始数据并保存为CSV
    processing_successful = process_marker_data_and_save(
        txt_filepath=txt_file_path,
        output_csv_path=output_csv_file
    )

    # 步骤2: 如果处理成功，则进行绘图
    if processing_successful:
        plot_pixel_coordinates_2d(pixel_csv_path=output_csv_file)
    else:
        print("\nData processing failed. Plotting will be skipped.")

    print("\nProgram finished.")