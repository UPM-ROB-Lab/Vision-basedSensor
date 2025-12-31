import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# 配置参数 (Configuration Parameters)
# =============================================================================

# --- 文件路径 ---
IMAGE_PATH = "markerdiameter/diameter1.jpg" # 您的图片文件名
OUTPUT_ANNOTATED_IMAGE_FILENAME = "annotated_verification_image.png"
OUTPUT_PLOT_FILENAME = "marker_diameter_summary.png"

# --- 标定板参数 ---
CHESSBOARD_PATTERN = (6, 6) 
SQUARE_SIZE_MM = 3.0

# --- 轮廓筛选参数 ---
MIN_MARKER_AREA = 100 
MIN_CIRCULARITY = 0.85

# --- 直径补偿值 (单位: mm) ---
DIAMETER_COMPENSATION_MM = 0


# =============================================================================
# 核心功能 (Core Functions)
# =============================================================================

def calculate_scale_from_chessboard(gray_image, pattern, square_size_mm):
    """从棋盘格计算图像的比例尺 (pixels/mm)"""
    print("Step 1: Finding chessboard to calculate scale...")
    found, corners = cv2.findChessboardCorners(gray_image, pattern, None)
    if not found:
        print("[ERROR] Chessboard corners not found. Cannot calculate scale.")
        return None, None
    print(f"-> Found {len(corners)} chessboard corners.")
    distances = []
    for r in range(pattern[1]):
        for c in range(pattern[0] - 1):
            dist = np.linalg.norm(corners[r * pattern[0] + c] - corners[r * pattern[0] + c + 1])
            distances.append(dist)
    for r in range(pattern[1] - 1):
        for c in range(pattern[0]):
            dist = np.linalg.norm(corners[r * pattern[0] + c] - corners[(r + 1) * pattern[0] + c])
            distances.append(dist)
    if not distances:
        print("[ERROR] Could not calculate distances between corners.")
        return None, None
    avg_dist_px = np.mean(distances)
    pixels_per_mm = avg_dist_px / square_size_mm
    print(f"-> Average square size in pixels: {avg_dist_px:.2f} px")
    print(f"-> Calculated Scale: {pixels_per_mm:.2f} pixels per mm")
    return pixels_per_mm, corners

def get_manual_threshold(gray_image):
    """
    打开一个带滑块的窗口，让用户手动选择最佳的全局阈值。
    """
    print("\n--- Manual Threshold Adjustment ---")
    print("Instructions:")
    print("1. A window named 'Threshold Adjuster' will appear.")
    print("2. Drag the slider to find a threshold where all markers are clear, solid white circles.")
    print("3. Once you are satisfied, press the 'ENTER' key to confirm.")
    print("4. If you close the window, the program will exit.")

    window_name = 'Threshold Adjuster'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    h, w = gray_image.shape
    new_width = int(w / 2)
    new_height = int(h / 2)
    cv2.resizeWindow(window_name, new_width, new_height)

    def nothing(x):
        pass
        
    cv2.createTrackbar('Threshold', window_name, 127, 255, nothing)
    
    while True:
        threshold_value = cv2.getTrackbarPos('Threshold', window_name)
        _, binary_img = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow(window_name, binary_img)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            threshold_value = -1
            break

    cv2.destroyAllWindows()
    
    if threshold_value != -1:
        print(f"-> Threshold confirmed at: {threshold_value}")
    else:
        print("-> Adjustment window closed. Exiting.")
        
    return threshold_value


def detect_and_measure_markers_by_contour(gray_image, scale_px_per_mm, threshold_value):
    """使用用户选择的阈值和轮廓检测来识别和测量marker。"""
    print("\nStep 2: Detecting and measuring markers with selected threshold...")
    _, thresh = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"-> Found {len(contours)} initial contours.")
    measured_diameters_mm = []
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_MARKER_AREA: continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY: continue
        (x, y), radius_px = cv2.minEnclosingCircle(cnt)
        diameter_px = radius_px * 2
        diameter_mm = diameter_px / scale_px_per_mm
        measured_diameters_mm.append(diameter_mm)
        valid_contours.append(cnt)
    print(f"-> Found {len(valid_contours)} valid markers after filtering.")
    if not valid_contours: return None, None
    return valid_contours, measured_diameters_mm


def visualize_and_get_annotated_image(original_image, corners, contours, diameters_mm):
    """将所有检测结果可视化并标注在图片上，同时返回这张标注图。"""
    print("\nStep 3: Visualizing results on original image...")
    output_image = original_image.copy()
    if corners is not None:
        cv2.drawChessboardCorners(output_image, CHESSBOARD_PATTERN, corners, True)
    if contours is not None:
        for i, cnt in enumerate(contours):
            cv2.drawContours(output_image, [cnt], -1, (0, 255, 0), 2)
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            diameter_mm = diameters_mm[i]
            text_pos = (center[0] - 20, center[1] - int(radius) - 10)
            cv2.putText(output_image, f"{diameter_mm:.2f}mm", text_pos, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    plt.figure(figsize=(18, 10))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title("Verification Results (Manual Threshold Method)", fontsize=20)
    plt.axis('off')
    plt.show()
    return output_image


def plot_diameter_summary(diameters_mm, target_diameter, save_path):
    """绘制一个柱状图来总结所有marker的直径测量结果。"""
    print("\nStep 5: Generating summary plot...")
    num_markers = len(diameters_mm)
    avg_diameter = np.mean(diameters_mm)
    std_dev = np.std(diameters_mm)
    marker_ids = np.arange(1, num_markers + 1)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(marker_ids, diameters_mm, color='c', edgecolor='black', label='Measured Diameter')
    ax.axhline(y=target_diameter, color='r', linestyle='--', linewidth=2, 
               label=f'Target Diameter ({target_diameter:.2f} mm)')
    text_str = (f'Markers Found: {num_markers}\n'
                f'Average Diameter: {avg_diameter:.2f} mm\n'
                f'Fluctuation: {avg_diameter:.2f} ± {std_dev:.2f} mm')
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7)
    ax.text(0.97, 0.97, text_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    ax.set_title('Marker Diameter Verification Summary', fontsize=16, weight='bold')
    ax.set_xlabel('Marker ID', fontsize=12)
    ax.set_ylabel('Measured Diameter (mm)', fontsize=12)
    ax.set_xticks(marker_ids)
    ax.legend()
    ax.set_ylim(bottom=0, top=max(diameters_mm) * 1.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"-> Summary plot saved to: '{save_path}'")
    plt.show()


def main():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"[FATAL] Image not found at '{IMAGE_PATH}'")
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    pixels_per_mm, chessboard_corners = calculate_scale_from_chessboard(gray, CHESSBOARD_PATTERN, SQUARE_SIZE_MM)
    if pixels_per_mm is None:
        return
        
    gray_blurred_for_markers = cv2.GaussianBlur(gray, (5, 5), 0)
    
    user_selected_threshold = get_manual_threshold(gray_blurred_for_markers)
    if user_selected_threshold == -1:
        return

    detected_contours, diameters_in_mm = detect_and_measure_markers_by_contour(
        gray_blurred_for_markers, pixels_per_mm, user_selected_threshold
    )
    
    if diameters_in_mm is None:
        print("[ERROR] Failed to find any valid markers with the selected threshold.")
        return
        
    # 【核心修改】对所有测量结果应用补偿值
    if DIAMETER_COMPENSATION_MM != 0:
        diameters_in_mm = (np.array(diameters_in_mm) + DIAMETER_COMPENSATION_MM).tolist()
        print(f"\n-> Applied a compensation of +{DIAMETER_COMPENSATION_MM} mm to all measured diameters.")

    annotated_image = visualize_and_get_annotated_image(image, chessboard_corners, detected_contours, diameters_in_mm)

    output_dir = os.path.dirname(IMAGE_PATH)
    if not output_dir: output_dir = "."
    
    annotated_image_save_path = os.path.join(output_dir, OUTPUT_ANNOTATED_IMAGE_FILENAME)
    cv2.imwrite(annotated_image_save_path, annotated_image)
    print(f"-> Annotated image saved to: '{annotated_image_save_path}'")

    print("\n--- Final Analysis ---")
    avg_diameter = np.mean(diameters_in_mm)
    std_dev = np.std(diameters_in_mm)
    min_diameter = np.min(diameters_in_mm)
    max_diameter = np.max(diameters_in_mm)
    print(f"Target Diameter: {SQUARE_SIZE_MM:.2f} mm")
    print(f"Average Measured Diameter: {avg_diameter:.2f} mm")
    print(f"Standard Deviation: {std_dev:.2f} mm")
    print(f"Diameter Range: [{min_diameter:.2f} mm, {max_diameter:.2f} mm]")
    
    plot_save_path = os.path.join(output_dir, OUTPUT_PLOT_FILENAME)
    plot_diameter_summary(diameters_in_mm, SQUARE_SIZE_MM, plot_save_path)


if __name__ == "__main__":
    main()