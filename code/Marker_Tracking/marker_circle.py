import cv2
import numpy as np
import setting
#import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy import ndimage
from scipy.signal import fftconvolve
import math
import pandas as pd 
import os 

def matching(N_, M_, fps_, x0_, y0_, dx_, dy_):

    N = N_;
    M = M_;
    NM = N * M;

    fps = fps_;

    x_0 = x0_; y_0 = y0_; dx = dx_; dy = dy_;

    Ox = np.zeros(NM)
    Oy = np.zeros(NM)
    for i in range(N):
        for j in range(M):
            Ox[i,j] = x_0 + j * dx;
            Oy[i,j] = y_0 + i * dy;

    flag_record = 1;

    dmin = (dx * 0.5) * (dx * 0.5);
    dmax = (dx * 1.8) * (dx * 1.8);
    theta = 70;
    moving_max = dx * 2;
    flow_difference_threshold = dx * 0.8;
    cost_threshold = 15000 * (dx / 21)* (dx / 21);


def gkern(l=5, sig=1.):
    """ creates gaussian kernel with side length l and a sigma of sig """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel / np.sum(kernel)

def normxcorr2(template, image, mode="same"):
    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")
    template = template - np.mean(template)
    image = image - np.mean(image)
    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)
    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))
    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0
    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)
    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0
    return out


def init(frame):
    RESCALE = setting.RESCALE
    return cv2.resize(frame, (0, 0), fx=1.0/RESCALE, fy=1.0/RESCALE)


def preprocessimg(img):
    '''
    Pre-processing image to remove noise
    '''
    dotspacepx = 36
    ### speckle noise denoising
    # dst = cv2.fastNlMeansDenoising(img_gray, None, 9, 15, 30)
    ### adaptive histogram equalizer
    # clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(10, 10))
    # equalized = clahe.apply(img_gray)
    ### Gaussian blur
    # gsz = 2 * round(3 * mindiameterpx / 2) + 1
    gsz = 2 * round(0.75 * dotspacepx / 2) + 1
    blur = cv2.GaussianBlur(img, (51, 51), gsz / 6)
    #### my linear varying filter
    x = np.linspace(3, 1.5, img.shape[1])
    y = np.linspace(3, 1.5, img.shape[0])
    xx, yy = np.meshgrid(x, y)
    mult = blur * yy
    ### adjust contrast
    res = cv2.convertScaleAbs(blur, alpha=2, beta=0)
    return res

# 这个函数用来为矫正畸变的marker识别
# def find_marker(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     im_blur_3 = cv2.GaussianBlur(gray, (39, 39), 8)
#     im_blur_8 = cv2.GaussianBlur(gray, (101, 101), 20)
#     im_blur_sub = im_blur_8 - im_blur_3 + 10
    
#     # 这是能反映真实面积的mask
#     area_mask = cv2.inRange(im_blur_sub, 30, 200)

#     # --- 后续步骤用于精确定位中心点 --- 
#     template = gkern(l=80, sig=13)
#     # 注意：模板匹配应该在 area_mask 上进行
#     nrmcrimg = normxcorr2(template, area_mask) 
    
#     a = nrmcrimg
#     # 这是用于精确定位质心的mask
#     mask = (a > 0.1).astype('uint8')

#     # 返回两个mask
#     return mask, area_mask 

def find_marker(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 640*480情况
    im_blur_3 = cv2.GaussianBlur(gray, (21, 21), 4.56)
    im_blur_8 = cv2.GaussianBlur(gray, (35, 35), 11.4)
    im_blur_sub = im_blur_8 - im_blur_3 + 15
    # 这是能反映真实面积的mask
    area_mask = cv2.inRange(im_blur_sub, 35, 180)
    # --- 后续步骤用于精确定位中心点 --- 
    template = gkern(l=33, sig=7.4)
    
    # # 1280*1024情况
    # im_blur_3 = cv2.GaussianBlur(gray, (39, 39), 8)
    # im_blur_8 = cv2.GaussianBlur(gray, (101, 101), 20)
    # im_blur_sub = im_blur_8 - im_blur_3 + 15
    # # 这是能反映真实面积的mask
    # area_mask = cv2.inRange(im_blur_sub, 20, 200)
    # # --- 后续步骤用于精确定位中心点 --- 
    # template = gkern(l=80, sig=13)
    
    
    # 注意：模板匹配应该在 area_mask 上进行
    nrmcrimg = normxcorr2(template, area_mask) 
    
    a = nrmcrimg
    # 这是用于精确定位质心的mask
    mask = (a > 0.1).astype('uint8')

    # 返回两个mask
    return mask, area_mask 

def marker_center(mask, area_mask, frame):
    """
    修改后的函数，添加距离阈值约束（匹配距离 < 椭圆半短轴的1/5），提升质心与椭圆的匹配准确性
    """
    img3 = mask.astype(np.uint8)
    # 640*480情况
    neighborhood_size = 8
    threshold = 0
    
    # 1280*1024情况（根据实际需求注释切换）
    # neighborhood_size = 14
    # threshold = 0
    
    # 提取局部最大值作为潜在质心（优化噪点过滤）
    data_max = maximum_filter(img3, neighborhood_size)
    maxima = (img3 == data_max)
    data_min = minimum_filter(img3, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    
    labeled, num_objects = ndimage.label(maxima)
    if num_objects == 0:
        return []
    
    # 计算质心（保持浮点精度）
    marker_centers_yx = np.array(ndimage.center_of_mass(img3, labeled, range(1, num_objects + 1)))
    if marker_centers_yx.ndim == 1 and num_objects == 1:
        marker_centers_yx = marker_centers_yx.reshape(1, -1)
    if marker_centers_yx.size == 0:
        return []
    
    # 轮廓提取（优化轮廓质量）
    if np.max(area_mask) > 1:
        area_mask_8u = area_mask.astype(np.uint8)
    else:
        area_mask_8u = (area_mask * 255).astype(np.uint8)
    
    # 对轮廓进行开运算，去除小噪点和毛刺
    kernel = np.ones((5, 5), np.uint8)
    area_mask_8u = cv2.morphologyEx(area_mask_8u, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(area_mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output_data = []
    
    # 质心转换为 (x, y) 格式
    centers_xy_list = [(c[1], c[0]) for c in marker_centers_yx] 
    unmatched_centers_with_idx = list(enumerate(centers_xy_list))  # 保留未匹配质心的索引和坐标

    # 基于距离阈值的匹配逻辑
    for contour in contours:
        # 过滤点数不足的轮廓（避免椭圆拟合失败）
        if len(contour) < 5:
            continue
            
        # 拟合椭圆并确定长短轴
        ellipse = cv2.fitEllipse(contour)
        (ellipse_cx, ellipse_cy), (width, height), angle_raw = ellipse
        
        # 明确长短轴（短轴用于计算阈值）
        if width > height:
            major_axis = width
            minor_axis = height
            angle = angle_raw
        else:
            major_axis = height
            minor_axis = width
            angle = angle_raw + 90
        
        # 过滤异常短轴（避免除以零或过小值）
        if minor_axis < 5:  # 假设最小短轴不小于5像素（可根据实际调整）
            continue
        
        # 计算距离阈值：必须小于椭圆半短轴的1/5（半短轴 = minor_axis/2，1/5即 minor_axis/(2*5) = minor_axis/10）
        distance_threshold = minor_axis / 10  # 允许的最大距离
        distance_threshold_sq = distance_threshold **2  # 平方形式，避免开方运算
        
        # 寻找轮廓内符合距离阈值的最佳质心
        best_match_idx_in_unmatched = -1
        min_distance_sq = float('inf')
        
        for i, (original_idx, center_xy) in enumerate(unmatched_centers_with_idx):
            cx, cy = center_xy
            
            # 条件1：质心必须在轮廓内
            if cv2.pointPolygonTest(contour, center_xy, False) >= 0:
                # 条件2：距离椭圆中心的距离必须小于阈值
                distance_sq = (cx - ellipse_cx)** 2 + (cy - ellipse_cy) **2
                
                if distance_sq < distance_threshold_sq and distance_sq < min_distance_sq:
                    min_distance_sq = distance_sq
                    best_match_idx_in_unmatched = i
        
        # 只有符合阈值的匹配才被接受
        if best_match_idx_in_unmatched != -1:
            # 移除匹配的质心，避免重复匹配
            original_idx, matched_center_xy = unmatched_centers_with_idx.pop(best_match_idx_in_unmatched)
            Cx, Cy = matched_center_xy
            
            output_data.append({
                'center': matched_center_xy, 
                'major_axis': float(major_axis),
                'minor_axis': float(minor_axis),
                'angle': float(angle)
            })
            
            # 可视化（保持原逻辑）
            if frame is not None:
                cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
                
                angle_rad = math.radians(angle)
                major_p1 = (Cx - major_axis/2 * math.cos(angle_rad), 
                           Cy - major_axis/2 * math.sin(angle_rad))
                major_p2 = (Cx + major_axis/2 * math.cos(angle_rad), 
                           Cy + major_axis/2 * math.sin(angle_rad))
                
                minor_angle_rad = angle_rad + math.pi/2
                minor_p1 = (Cx - minor_axis/2 * math.cos(minor_angle_rad), 
                           Cy - minor_axis/2 * math.sin(minor_angle_rad))
                minor_p2 = (Cx + minor_axis/2 * math.cos(minor_angle_rad), 
                           Cy + minor_axis/2 * math.sin(minor_angle_rad))
                
                cv2.line(frame, (int(major_p1[0]), int(major_p1[1])),
                        (int(major_p2[0]), int(major_p2[1])), (0, 255, 255), 2)
                cv2.line(frame, (int(minor_p1[0]), int(minor_p1[1])),
                        (int(minor_p2[0]), int(minor_p2[1])), (255, 0, 0), 2)
    
    return output_data


def draw_flow(frame, flow):
    Ox, Oy, Cx, Cy, Occupied = flow

    dx = np.mean(np.abs(np.asarray(Ox) - np.asarray(Cx)))
    dy = np.mean(np.abs(np.asarray(Oy) - np.asarray(Cy)))
    dnet = np.sqrt(dx**2 + dy**2)
    print (dnet * 0.075, '\n')


    K = 1
    for i in range(len(Ox)):
        for j in range(len(Ox[i])):
            pt1 = (int(Ox[i][j]), int(Oy[i][j]))
            pt2 = (int(Cx[i][j] + K * (Cx[i][j] - Ox[i][j])), int(Cy[i][j] + K * (Cy[i][j] - Oy[i][j])))
            color = (0, 0, 255)
            if Occupied[i][j] <= -1:
                color = (127, 127, 255)
            cv2.arrowedLine(frame, pt1, pt2, color, 2,  tipLength=0.25)


def warp_perspective(img): # 透视变换

    TOPLEFT = (175,230)
    TOPRIGHT = (380,225)
    BOTTOMLEFT = (10,410)
    BOTTOMRIGHT = (530,400)

    WARP_W = 215
    WARP_H = 215

    points1=np.float32([TOPLEFT,TOPRIGHT,BOTTOMLEFT,BOTTOMRIGHT])
    points2=np.float32([[0,0],[WARP_W,0],[0,WARP_H],[WARP_W,WARP_H]])

    matrix=cv2.getPerspectiveTransform(points1,points2)

    result = cv2.warpPerspective(img, matrix, (WARP_W,WARP_H))

    return result



def save_new_intrinsics_to_excel(new_camera_matrix, filepath): # 保存新内参矩阵到Excel文件
    """
    将计算出的新相机内参矩阵保存到指定的Excel文件。

    Args:
        new_camera_matrix (np.ndarray): 3x3 的新内参矩阵。
        filepath (str): 输出的Excel文件路径。
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 从矩阵中提取参数
    new_fx = new_camera_matrix[0, 0]
    new_fy = new_camera_matrix[1, 1]
    new_cx = new_camera_matrix[0, 2]
    new_cy = new_camera_matrix[1, 2]

    # 准备要写入的数据
    data = {
        'Parameter': ['new_fx', 'new_fy', 'new_cx', 'new_cy'],
        'Value': [new_fx, new_fy, new_cx, new_cy],
        'Description': [
            'Focal length in x-axis for undistorted image (pixels)',
            'Focal length in y-axis for undistorted image (pixels)',
            'Principal point x-coordinate for undistorted image (pixels)',
            'Principal point y-coordinate for undistorted image (pixels)'
        ]
    }

    # 创建DataFrame并保存
    df = pd.DataFrame(data)
    try:
        df.to_excel(filepath, index=False, engine='openpyxl')
        print(f"\nSuccessfully saved new intrinsic parameters to: '{filepath}'")
    except Exception as e:
        print(f"\nERROR: Failed to save new intrinsic parameters to Excel. Reason: {e}")


def init_HSR(img):
    # 1. 定义目标分辨率
    TARGET_WIDTH = 1280
    TARGET_HEIGHT = 1024
    TARGET_DIM = (TARGET_WIDTH, TARGET_HEIGHT)
    
    # 2. 【已修改】检查输入图像的尺寸
    h_input, w_input = img.shape[:2]
    if (w_input, h_input) != TARGET_DIM:
        # 如果尺寸不匹配，则引发一个错误，而不是进行缩放
        raise ValueError(f"输入图像尺寸 ({w_input}, {h_input}) 与预期的标定尺寸 {TARGET_DIM} 不匹配。请提供正确分辨率的图像。")
    
    # 如果尺寸匹配，程序将继续执行
    # 从这里开始，img_resized 变量可以被 img 替代，因为我们已经确保了尺寸正确
    
    # 3. 加载标定参数
    # 相机内参矩阵 K (针对 1280x1024 分辨率)
    '''
            camera_matrix = np.array([
            [params['fx'], skew,         params['cx']],
            [0,            params['fy'], params['cy']],
            [0,            0,            1]
        ], dtype=np.float32)

        # 构建畸变系数数组 D
        dist_keys = ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6']
    '''

    # 相机内参矩阵 K (使用OpenCV标准格式)
    K_calibrated = np.array([
        [1793.502947,    0.0,         627.7660117],
        [   0.0,      1852.128659,   418.7903304],
        [   0.0,         0.0,           1.0      ]
    ], dtype=np.float32)

# 畸变系数 D = [k1, k2, p1, p2, k3]
    D_calibrated = np.array([-0.339501755, 0.27796056, -0.006545208, -0.034617897, 0.875068688], dtype=np.float32)

    # 4. 计算优化的相机矩阵 (与之前相同)
    img_size = TARGET_DIM
    alpha = 0
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K_calibrated, D_calibrated, img_size, alpha, img_size)

    # 5. 计算映射表 (与之前相同)
    map1, map2 = cv2.initUndistortRectifyMap(
        K_calibrated,
        D_calibrated,
        None,
        new_camera_matrix,
        img_size,
        cv2.CV_16SC2
    )

    # 6. 应用映射表进行重映射 (与之前相同)
    # 注意：这里直接使用 img，因为我们已经确认了它的尺寸是正确的
    undistorted_img = cv2.remap(
        img,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR
    )
    
    return undistorted_img, new_camera_matrix

