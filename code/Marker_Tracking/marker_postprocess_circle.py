import cv2
import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from marker_circle import find_marker, marker_center 
import math

def process_video():
    """自动处理视频，并将数据保存到CSV，其中Ox,Oy为第一帧的初始位置。"""
    
    # 输入输出路径配置
    video_dir = './video'
    input_path = os.path.join(video_dir, 'test2.avi')
    output_video_path = os.path.join(video_dir, 'exp_processed.avi')
    output_csv_path = os.path.join(video_dir, 'marker_locations_0.csv')

    # 使用与照片处理完全相同的裁剪参数 (左, 右, 上, 下)
    # crop_ratios = (1/8, 1/8, 0, 1/16)  # 5*5阵列裁剪系数
    crop_ratios = (1/8, 1/8, 1/16, 0) # 环形阵列裁剪系数
    
    # 确保视频目录存在
    os.makedirs(video_dir, exist_ok=True)
    
    # 初始化视频捕获
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件: {input_path}")
    
    # 获取原始视频参数
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 计算裁剪后的尺寸
    left = int(width * crop_ratios[0])
    right = width - int(width * crop_ratios[1])
    top = int(height * crop_ratios[2])
    bottom = height - int(height * crop_ratios[3])
    crop_width = right - left
    crop_height = bottom - top
    
    # 初始化视频写入器 (使用裁剪后的尺寸)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (crop_width, crop_height))
    
    # 数据存储结构，保持 'row' 和 'col' 列名不变
    data = {
        'frameno': [],
        'row': [], # 存储 layer_idx
        'col': [], # 存储 angle_idx
        'Ox': [],
        'Oy': [],
        'Cx': [],
        'Cy': [],
        'major_axis': [],
        'minor_axis': [],
        'angle': []
    }
    
    first_frame_markers_indexed = {} # 存储第一帧标记点，以 (layer_idx, angle_idx) 为键
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cropped_frame = frame[top:bottom, left:right]
        # 调用导入的 find_marker 和 marker_center
        mask, area_mask = find_marker(cropped_frame)
        markers = marker_center(mask, area_mask, cropped_frame)
        
        # 复制帧用于绘制结果
        result_frame = cropped_frame.copy()
        
        if frame_count == 0:
            # --- 新的圆环形排列身份建立逻辑 ---
            if not markers:
                print("错误：第一帧未检测到任何标记点。请确保视频第一帧图像清晰。")
                break
            
            marker_centers_xy = np.array([m['center'] for m in markers])
            
            # 1. 识别最中心的标记点
            geometric_mean_center_x = np.mean(marker_centers_xy[:, 0])
            geometric_mean_center_y = np.mean(marker_centers_xy[:, 1])
            geometric_mean_center = np.array([geometric_mean_center_x, geometric_mean_center_y])
            
            distances_to_mean_center = np.linalg.norm(marker_centers_xy - geometric_mean_center, axis=1)
            center_marker_idx_in_all = np.argmin(distances_to_mean_center)
            
            center_marker_data = markers[center_marker_idx_in_all]
            actual_array_center = np.array(center_marker_data['center']) 
            
            remaining_markers = [m for i, m in enumerate(markers) if i != center_marker_idx_in_all]
            
            # 2. 处理中心标记点：分配 (0, 0) ID
            first_frame_markers_indexed[(0, 0)] = { # layer_idx=0, angle_idx=0 代表中心点
                'Ox': actual_array_center[0],
                'Oy': actual_array_center[1],
                'major_axis': center_marker_data['major_axis'],
                'minor_axis': center_marker_data['minor_axis'],
                'angle': center_marker_data['angle']
            }
            
            if not remaining_markers:
                # 如果只有中心点，则跳过后续分层，但确保中心点数据被记录
                # (此处省略了中心点数据记录和绘图，因为它们会在后续的通用跟踪循环中处理)
                pass
            else:
                # 3. 计算剩余标记点到实际中心点的极坐标 (半径和角度)
                remaining_centers_xy = np.array([m['center'] for m in remaining_markers])
                distances = np.linalg.norm(remaining_centers_xy - actual_array_center, axis=1)
                
                # np.arctan2(dy, dx) 返回 (-pi, pi] 范围内的角度
                angles = np.arctan2(remaining_centers_xy[:, 1] - actual_array_center[1], 
                                    remaining_centers_xy[:, 0] - actual_array_center[0]) 
                
                # 4. 使用 K-Means 聚类进行分层
                num_outer_layers = 5 # 有5个外围层
                
                distances_reshaped = distances.reshape(-1, 1)
                kmeans = KMeans(n_clusters=num_outer_layers, random_state=0, n_init=10) 
                kmeans.fit(distances_reshaped)
                
                # 确保 layer_idx 随着半径增大而增大
                sorted_cluster_centers = np.sort(kmeans.cluster_centers_.flatten())
                
                layer_assignments = np.zeros(len(remaining_markers), dtype=int)
                for i, dist_val in enumerate(distances):
                    closest_center_idx = np.argmin(np.abs(dist_val - kmeans.cluster_centers_.flatten()))
                    final_layer_idx = np.where(sorted_cluster_centers == kmeans.cluster_centers_.flatten()[closest_center_idx])[0][0]
                    layer_assignments[i] = final_layer_idx + 1 # +1 是因为中心点是 layer 0

                num_total_layers = num_outer_layers + 1 
                
                # 5. 在每个层内按角度排序，并分配唯一ID (实现循环移位)
                layers_data = [[] for _ in range(num_total_layers)] 
                for i, marker in enumerate(remaining_markers):
                    layer_idx = layer_assignments[i]
                    layers_data[layer_idx].append({
                        'center': marker['center'],
                        'major_axis': marker['major_axis'],
                        'minor_axis': marker['minor_axis'],
                        'angle': marker['angle'],
                        'angle_rad': angles[i]
                    })
                
                for layer_idx in range(1, num_total_layers): # 从 layer_idx=1 开始处理外围层
                    
                    current_layer_markers = layers_data[layer_idx]
                    if not current_layer_markers:
                        continue

                    # a. 按角度排序 (从 -pi 到 pi)
                    current_layer_markers.sort(key=lambda m: m['angle_rad'])

                    # b. 找到最接近 0 角度的点 (即最右侧的点)
                    min_angle_abs = np.inf
                    start_idx = 0
                    for i, marker_in_layer in enumerate(current_layer_markers):
                        # 找到角度绝对值最小的，即最接近正X轴的点
                        if abs(marker_in_layer['angle_rad']) < min_angle_abs:
                            min_angle_abs = abs(marker_in_layer['angle_rad'])
                            start_idx = i

                    # c. 执行循环移位，使 start_idx 成为新的 angle_idx = 0
                    shifted_list = current_layer_markers[start_idx:] + current_layer_markers[:start_idx]

                    # d. 分配新的 angle_idx (逆时针方向)
                    for angle_idx, marker_in_layer in enumerate(shifted_list):
                        # 存储到 first_frame_markers_indexed
                        first_frame_markers_indexed[(layer_idx, angle_idx)] = {
                            'Ox': marker_in_layer['center'][0],
                            'Oy': marker_in_layer['center'][1],
                            'major_axis': marker_in_layer['major_axis'],
                            'minor_axis': marker_in_layer['minor_axis'],
                            'angle': marker_in_layer['angle']
                        }
            
            # --- 结束身份建立逻辑 ---
        
        if first_frame_markers_indexed and markers:
            # 建立当前帧标记点的查找表 (中心坐标 -> 完整数据)
            current_markers_map = {tuple(m['center']): m for m in markers}
            current_centers_list = np.array([m['center'] for m in markers]) 
            
            # 遍历第一帧的网格化标记点 (现在是按层和角度索引)
            for (layer_idx, angle_idx), initial_marker_data in first_frame_markers_indexed.items():
                Ox, Oy = initial_marker_data['Ox'], initial_marker_data['Oy']
                
                if len(current_centers_list) == 0: 
                    continue
                
                # 寻找距离第一帧标记点最近的当前帧标记点
                distances_to_current = cdist(np.array([[Ox, Oy]]), current_centers_list)
                closest_idx = np.argmin(distances_to_current)
                best_match_center_tuple = tuple(current_centers_list[closest_idx])
                
                matched_data = current_markers_map.get(best_match_center_tuple)
                
                if matched_data:
                    Cx, Cy = matched_data['center']
                    major_axis = matched_data['major_axis']
                    minor_axis = matched_data['minor_axis']
                    angle = matched_data['angle']
                    
                    # 存储数据
                    data['frameno'].append(frame_count)
                    data['row'].append(layer_idx) # 存储 layer_idx 到 'row'
                    data['col'].append(angle_idx) # 存储 angle_idx 到 'col'
                    data['Ox'].append(Ox)
                    data['Oy'].append(Oy)
                    data['Cx'].append(Cx)
                    data['Cy'].append(Cy)
                    data['major_axis'].append(major_axis)
                    data['minor_axis'].append(minor_axis)
                    data['angle'].append(angle)

                    # 绘制部分
                    cv2.circle(result_frame, (int(Cx), int(Cy)), 4, (0, 0, 255), -1)
                    cv2.arrowedLine(result_frame, (int(Ox), int(Oy)), (int(Cx), int(Cy)), (0, 0, 255), 2, tipLength=0.25)

                    # 绘制长短轴
                    cv2.ellipse(result_frame, 
                                ((int(Cx), int(Cy)), (int(major_axis), int(minor_axis)), angle),
                                (0, 255, 0), 1)
                    
                    # 计算轴端点并绘制
                    angle_rad = np.deg2rad(angle)
                    major_p1 = (Cx - major_axis/2 * np.cos(angle_rad), Cy - major_axis/2 * np.sin(angle_rad))
                    major_p2 = (Cx + major_axis/2 * np.cos(angle_rad), Cy + major_axis/2 * np.sin(angle_rad))
                    
                    minor_angle_rad = angle_rad + np.pi/2
                    minor_p1 = (Cx - minor_axis/2 * np.cos(minor_angle_rad), Cy - minor_axis/2 * np.sin(minor_angle_rad))
                    minor_p2 = (Cx + minor_axis/2 * np.cos(minor_angle_rad), Cy + minor_axis/2 * np.sin(minor_angle_rad))
                    
                    cv2.line(result_frame, (int(major_p1[0]), int(major_p1[1])), (int(major_p2[0]), int(major_p2[1])), (0, 255, 255), 2)
                    cv2.line(result_frame, (int(minor_p1[0]), int(minor_p1[1])), (int(minor_p2[0]), int(minor_p2[1])), (255, 0, 0), 2)
        
        # 写入处理后的帧
        out.write(result_frame)
        frame_count += 1
    
    # 释放资源
    cap.release()
    out.release()
    
    # 保存CSV数据
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    
    print(f"处理完成！结果已保存到: {video_dir}")

if __name__ == '__main__':
    process_video()