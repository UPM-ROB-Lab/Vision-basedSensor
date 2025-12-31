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
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs.
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """
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

# def find_marker(frame):
#     # gray = frame[:,:,1] ### use only the green channel
#     # im_blur_3 = cv2.GaussianBlur(gray,(3,3),5)
#     # im_blur_8 = cv2.GaussianBlur(gray, (15,15),5)
#     # im_blur_sub = im_blur_8 - im_blur_3 + 128
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # gray = cv2.bitwise_not(gray_temp) # 反转图像，使白色marker变暗
#     im_blur_3 = cv2.GaussianBlur(gray,(39,39),8) # 选取D/2-D之间即可
#     im_blur_8 = cv2.GaussianBlur(gray, (101,101),20) # 要大于直径
#     im_blur_sub = im_blur_8 - im_blur_3 + 10 # 加的越多，暗色部分越亮
#     # 在 find_marker 中，计算完 im_blur_sub 后
#     mask = cv2.inRange(im_blur_sub, 30, 200) # 提取亮的地方作为marker点
#     # mask = cv2.inRange(im_blur_sub, 0, 200) # 越低，对边缘提取效果越好

#     # ''' normalized cross correlation '''
#     # template = gkern(l=20, sig=3) # l要近似与marker点的直径
#     template = gkern(l=80, sig=13) # l越大，找到的marker点越大
#     nrmcrimg = normxcorr2(template, mask)
#     # ''''''''''''''''''''''''''''''''''''
#     a = nrmcrimg
#     mask = np.asarray(a > 0.1)
#     mask = (mask).astype('uint8')

#     return mask

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
    im_blur_3 = cv2.GaussianBlur(gray, (39, 39), 8)
    im_blur_8 = cv2.GaussianBlur(gray, (101, 101), 20)
    im_blur_sub = im_blur_8 - im_blur_3 + 10
    
    # 这是能反映真实面积的mask
    area_mask = cv2.inRange(im_blur_sub, 20, 200)

    # --- 后续步骤用于精确定位中心点 --- 
    template = gkern(l=80, sig=13)
    # 注意：模板匹配应该在 area_mask 上进行
    nrmcrimg = normxcorr2(template, area_mask) 
    
    a = nrmcrimg
    # 这是用于精确定位质心的mask
    mask = (a > 0.1).astype('uint8')

    # 返回两个mask
    return mask, area_mask 

def marker_center(mask, area_mask, frame):
    """
    Finds marker centers, fits ellipses, and calculates the TRUE major axis,
    minor axis, and the major axis angle. Also visualizes both axes.
    """
    # Step 1 & 2: (代码未改变, 保持原样)
    img3 = mask.astype(np.uint8)
    neighborhood_size = 14
    threshold = 0
    data_max = maximum_filter(img3, neighborhood_size)
    maxima = (img3 == data_max)
    data_min = minimum_filter(img3, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    labeled, num_objects = ndimage.label(maxima)

    if num_objects == 0:
        return np.array([]).reshape(0, 2), np.array([]), np.array([]), np.array([])

    marker_centers_yx = np.array(ndimage.center_of_mass(img3, labeled, range(1, num_objects + 1)))

    if marker_centers_yx.ndim == 1 and num_objects == 1:
        marker_centers_yx = marker_centers_yx.reshape(1, -1)
    if marker_centers_yx.size == 0 and num_objects > 0:
        return np.array([]).reshape(0, 2), np.array([]), np.array([]), np.array([])

    if np.max(area_mask) > 1:
        area_mask_8u = (area_mask).astype(np.uint8)
    else:
        area_mask_8u = (area_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(area_mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 3: Associate centers, calculate axes and angle, and DRAW
    major_axes_lengths = []
    minor_axes_lengths = [] 
    major_axis_angles = [] 
    valid_centers_yx = []
    centers_list = list(marker_centers_yx)

    for contour in contours:
        if len(contour) < 5:
            continue
        ellipse = cv2.fitEllipse(contour)
        
        found_center_idx = -1
        for i, center_yx in enumerate(centers_list):
            center_xy = (center_yx[1], center_yx[0])
            if cv2.pointPolygonTest(contour, center_xy, False) >= 0:
                found_center_idx = i
                break
        
        if found_center_idx != -1:
            (cx, cy), (width, height), angle_raw = ellipse

            if width > height:
                major_axis_len = width
                minor_axis_len = height
                corrected_angle = angle_raw
            else:
                major_axis_len = height
                minor_axis_len = width
                corrected_angle = angle_raw + 90

            major_axes_lengths.append(major_axis_len)
            minor_axes_lengths.append(minor_axis_len)
            major_axis_angles.append(corrected_angle)
            
            valid_centers_yx.append(centers_list[found_center_idx])
            del centers_list[found_center_idx]

            # --- 【可视化部分已修改】 ---
            if frame is not None:
                # 1. 绘制绿色椭圆轮廓 (不变)
                cv2.ellipse(frame, ellipse, (0, 255, 0), 2)

                # 2. 绘制黄色的长轴
                angle_rad_major = math.radians(corrected_angle)
                half_major_axis = major_axis_len / 2.0
                
                dx_major = half_major_axis * math.cos(angle_rad_major)
                dy_major = half_major_axis * math.sin(angle_rad_major)
                
                p1_major = (int(cx - dx_major), int(cy - dy_major))
                p2_major = (int(cx + dx_major), int(cy + dy_major))
                
                cv2.line(frame, p1_major, p2_major, color=(0, 255, 255), thickness=2) # 黄色

                # 3. 【新增】绘制蓝色的短轴
                # 短轴与长轴垂直，所以其角度是长轴角度+90度
                angle_rad_minor = math.radians(corrected_angle + 90)
                half_minor_axis = minor_axis_len / 2.0

                dx_minor = half_minor_axis * math.cos(angle_rad_minor)
                dy_minor = half_minor_axis * math.sin(angle_rad_minor)

                p1_minor = (int(cx - dx_minor), int(cy - dy_minor))
                p2_minor = (int(cx + dx_minor), int(cy + dy_minor))

                cv2.line(frame, p1_minor, p2_minor, color=(255, 0, 0), thickness=2) # 蓝色
            # --- 【可视化修改结束】 ---

    # 【修改】如果没有任何有效的中心点，返回一个空列表
    if not valid_centers_yx:
        return [] # 返回空列表

    # 【修改】将所有数据打包成一个字典列表
    output_data = []
    for i in range(len(valid_centers_yx)):
        center_yx = valid_centers_yx[i]
        output_data.append({
            'center': (center_yx[1], center_yx[0]), # (x, y) 格式
            'major_axis': major_axes_lengths[i],
            'minor_axis': minor_axes_lengths[i],
            'angle': major_axis_angles[i]
        })

    return output_data


# def marker_center(mask, area_mask, frame):
#     # Step 1: Use mask to find precise locations (peaks and centroids)
#     # Ensure the input is a uint8 array for processing
#     img3 = mask.astype(np.uint8)

#     # Define parameters for peak finding
#     neighborhood_size = 14
#     threshold = 0  # for mini

#     # Find local maxima which will be our candidate marker centers
#     data_max = maximum_filter(img3, neighborhood_size)
#     maxima = (img3 == data_max)
#     data_min = minimum_filter(img3, neighborhood_size)
#     diff = ((data_max - data_min) > threshold)
#     maxima[diff == 0] = 0

#     # Label each independent peak (marker) with a unique integer
#     labeled, num_objects = ndimage.label(maxima)

#     # If no objects are detected, return empty arrays immediately
#     if num_objects == 0:
#         return np.array([]).reshape(0, 2), np.array([])

#     # Calculate the center of mass for each labeled region in img3 (center_mask)
#     # This gives us the precise (y, x) coordinates of the centroids
#     marker_centers_yx = np.array(ndimage.center_of_mass(img3, labeled, range(1, num_objects + 1)))

#     # Handle cases where only one object is found or the result is empty
#     if marker_centers_yx.ndim == 1 and num_objects == 1:
#         marker_centers_yx = marker_centers_yx.reshape(1, -1)
#     if marker_centers_yx.size == 0 and num_objects > 0:
#         return np.array([]).reshape(0, 2), np.array([])

#     # Step 2: Use area_mask to calculate the area of each marker
#     # The `labeled` map, generated from center_mask, will now be used to define regions on the area_mask.
    
#     # Prepare the area_mask for summation (ensure it's a 0-1 binary image)
#     if np.max(area_mask) == 255:
#         mask_for_area = (area_mask // 255).astype(np.uint8)
#     else:
#         mask_for_area = area_mask.astype(np.uint8)

#     # Sum the pixels in `mask_for_area` for each region defined by `labeled`.
#     # This effectively counts the number of white pixels in area_mask for each detected marker.
#     areas = ndimage.sum(mask_for_area, labeled, index=range(1, num_objects + 1))

#     # Step 3: Finalize the output
#     # Convert the center coordinates from (y, x) to (x, y) format
#     MarkerCenter = marker_centers_yx[:, [1, 0]]

#     # (Optional) Draw circles on the original frame for visualization
#     if frame is not None:
#         for i in range(MarkerCenter.shape[0]):
#             x0, y0 = int(MarkerCenter[i, 0]), int(MarkerCenter[i, 1])
#             # Draw on the area_mask to see what is being measured
#             cv2.circle(area_mask, (x0, y0), color=(0, 0, 0), radius=3, thickness=-1)

#     return MarkerCenter, areas

# def marker_center(mask, frame): # frame 参数用于绘制，如果不需要绘制可以移除
#     img3 = mask.astype(np.uint8) # 确保 mask 是 uint8 类型

#     neighborhood_size = 14
#     threshold = 0 # for mini

#     data_max = maximum_filter(img3, neighborhood_size)
#     maxima = (img3 == data_max)
#     data_min = minimum_filter(img3, neighborhood_size)
#     diff = ((data_max - data_min) > threshold)
#     maxima[diff == 0] = 0

#     labeled, num_objects = ndimage.label(maxima)

#     # 如果没有检测到对象，直接返回空的 MarkerCenter 和 areas
#     if num_objects == 0:
#         return np.array([]).reshape(0, 2), np.array([]) #确保MarkerCenter是0x2的形状

#     # 计算质心 (y,x)
#     marker_centers_yx = np.array(ndimage.center_of_mass(img3, labeled, range(1, num_objects + 1)))

#     # 如果只有一个对象，ndimage.center_of_mass 返回一维数组，需要 reshape
#     if marker_centers_yx.ndim == 1 and num_objects == 1:
#         marker_centers_yx = marker_centers_yx.reshape(1, -1)
#     if marker_centers_yx.size == 0 and num_objects > 0:
#         return np.array([]).reshape(0, 2), np.array([])

#     # 计算每个标签区域的面积 (像素数量)
#     if np.max(img3) == 255:
#         mask_for_area = (img3 // 255).astype(np.uint8)
#     else:
#         mask_for_area = img3.astype(np.uint8)

#     areas = ndimage.sum(mask_for_area, labeled, index=range(1, num_objects + 1))
#     # 确保 areas 和 marker_centers_yx 的长度一致，如果 ndimage.sum 由于某些原因

#     # 转换中心点坐标顺序 (y,x) -> (x,y) 并命名为 MarkerCenter
#     MarkerCenter = marker_centers_yx[:, [1, 0]] # 这就是你想要的 MarkerCenter

#     # (可选) 在传入的 frame (或 mask) 上绘制中心点
#     if frame is not None:
#         for i in range(MarkerCenter.shape[0]):
#             x0, y0 = int(MarkerCenter[i, 0]), int(MarkerCenter[i, 1])
#             cv2.circle(mask, (x0, y0), color=(0, 0, 0), radius=1, thickness=1)

#     return MarkerCenter, areas


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

