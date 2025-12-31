import cv2
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.signal import fftconvolve
import math
import os

def gkern(l=5, sig=1.):
    """创建高斯核"""
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel / np.sum(kernel)

def crop_borders(image, crop_ratios=(1/8, 1/8, 1/8, 1/8)):
    """
    按比例裁剪图像边框（四个方向独立设置）
    :param image: 输入图像
    :param crop_ratios: 裁剪比例元组 (左, 右, 上, 下)
    :return: 裁剪后的图像
    """
    h, w = image.shape[:2]
    left_ratio, right_ratio, top_ratio, bottom_ratio = crop_ratios
    
    # 计算各边裁剪像素数
    left = int(w * left_ratio)
    right = w - int(w * right_ratio)
    top = int(h * top_ratio)
    bottom = h - int(h * bottom_ratio)
    
    # 有效性检查
    if right <= left or bottom <= top:
        raise ValueError("裁剪比例过大，会导致图像尺寸无效")
    
    # 执行裁剪 (y: top->bottom, x: left->right)
    cropped = image[top:bottom, left:right]
    return cropped

def process_image_debug(img_path, params):
    """
    调试用图像处理全流程
    :param img_path: 输入图像路径
    :param params: 参数字典，包含所有可调参数
    :return: 并排显示的对比图
    """
    # 读取图像
    img_original = cv2.imread(img_path)
    if img_original is None:
        raise ValueError(f"无法加载图像，请检查路径: {img_path}")
    
    # 裁剪图像边框
    try:
        img_original = crop_borders(img_original, params.get('crop_ratios', (1/8, 1/8, 1/8, 1/8)))
    except ValueError as e:
        print(f"警告: {str(e)}, 跳过裁剪步骤")
    
    # 复制用于处理的图像
    debug_img = img_original.copy()
    
    # === 完整处理流程 ===
    # 1. 灰度转换
    gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    
    # 2. 双重高斯滤波
    im_blur_3 = cv2.GaussianBlur(gray, params['blur_small'], params['sigma_small'])
    im_blur_8 = cv2.GaussianBlur(gray, params['blur_large'], params['sigma_large'])
    
    # 3. 背景减除
    im_blur_sub = cv2.subtract(im_blur_8, im_blur_3)
    im_blur_sub = cv2.add(im_blur_sub, params['brightness_offset'])
    
    # 4. 阈值处理
    area_mask = cv2.inRange(im_blur_sub, params['thresh_low'], params['thresh_high'])
    area_mask_8u = (area_mask // 255).astype(np.uint8)
    
    # 5. 模板匹配
    template = gkern(l=params['template_size'], sig=params['template_sigma'])
    nrmcrimg = normxcorr2(template, area_mask_8u)
    center_mask = (nrmcrimg > params['ncr_threshold']).astype(np.uint8)
    
    # 6. 峰值检测
    data_max = maximum_filter(center_mask, params['neighborhood_size'])
    maxima = (center_mask == data_max)
    data_min = minimum_filter(center_mask, params['neighborhood_size'])
    diff = ((data_max - data_min) > 0)
    maxima[diff == 0] = 0
    
    # 7. 椭圆拟合与绘制
    contours, _ = cv2.findContours(area_mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if len(contour) < 5:
            continue
            
        # 椭圆拟合
        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (width, height), angle = ellipse
        
        # 半径补偿
        width *= params['radius_comp']
        height *= params['radius_comp']
        ellipse = ((cx, cy), (width, height), angle)
        
        # 绘制椭圆和轴线
        cv2.ellipse(debug_img, ellipse, (0, 255, 0), 2)
        
        # 计算长轴端点
        angle_rad = math.radians(angle)
        half_major = max(width, height) / 2
        dx = half_major * math.cos(angle_rad)
        dy = half_major * math.sin(angle_rad)
        
        # 绘制长轴(黄色)
        cv2.line(debug_img, 
                (int(cx - dx), int(cy - dy)),
                (int(cx + dx), int(cy + dy)),
                (0, 255, 255), 2)
        
        # 绘制短轴(蓝色)
        half_minor = min(width, height) / 2
        dx = half_minor * math.cos(angle_rad + math.pi/2)
        dy = half_minor * math.sin(angle_rad + math.pi/2)
        cv2.line(debug_img,
                (int(cx - dx), int(cy - dy)),
                (int(cx + dx), int(cy + dy)),
                (255, 0, 0), 2)
    
    # 合并显示
    result = np.hstack([img_original, debug_img])
    
    # 添加参数信息
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_pos = 30
    for key, value in params.items():
        if key == 'crop_ratios':  # 特殊格式化显示裁剪比例
            cv2.putText(result, f"{key}: (左{value[0]:.2f}, 右{value[1]:.2f}, 上{value[2]:.2f}, 下{value[3]:.2f})", 
                       (10, y_pos), font, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(result, f"{key}: {value}", (10, y_pos), 
                       font, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        y_pos += 30
    
    # 显示分辨率信息
    h, w = img_original.shape[:2]
    cv2.putText(result, f"Original: {w}x{h} (已裁剪)", (w+10, 30), 
               font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    
    return result

def normxcorr2(template, image):
    """归一化互相关"""
    template = template - np.mean(template)
    image = image - np.mean(image)
    a1 = np.ones(template.shape)
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode='same')
    image = fftconvolve(np.square(image), a1, mode='same') - \
            np.square(fftconvolve(image, a1, mode='same')) / np.prod(template.shape)
    image = np.maximum(image, 0)
    out = out / np.sqrt(image * np.sum(np.square(template)))
    out[np.where(~np.isfinite(out))] = 0
    return out

# 预设参数 (基于40px半径优化)
DEFAULT_PARAMS = {
    'blur_small': (41, 41),
    'blur_large': (59, 59),
    'sigma_small': 4.56,
    'sigma_large': 11.4,
    'brightness_offset': 15,
    'thresh_low': 45,
    'thresh_high': 200,
    'template_size': 46,
    'template_sigma': 7.4,
    'ncr_threshold': 0.1,
    'neighborhood_size': 8,
    'radius_comp': 1.075,
    'crop_ratios': (1/8,1/8,0,1/16)  # 修改为四元组格式 (左,右,上,下)
}

def main():
    # 确保video目录存在
    os.makedirs("video", exist_ok=True)
    
    # 输入输出路径
    input_path = "video/2.jpg"
    output_path = "video/output.jpg"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误：输入文件 {input_path} 不存在")
        return
    
    # 处理图像
    try:
        result = process_image_debug(input_path, DEFAULT_PARAMS)
        
        # 保存结果
        cv2.imwrite(output_path, result)
        print(f"处理完成！结果已保存到: {output_path}")
        
        # 显示结果 (自动调整窗口大小)
        h, w = result.shape[:2]
        display_width = min(1600, w)
        display_height = int(h * (display_width / w))
        
        cv2.namedWindow('调试结果', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('调试结果', display_width, display_height)
        cv2.imshow('调试结果', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")

if __name__ == "__main__":
    main()