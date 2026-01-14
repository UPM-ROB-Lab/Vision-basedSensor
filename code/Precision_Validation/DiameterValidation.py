"""
Marker Diameter Verification Tool

This script calculates the pixel-to-millimeter scale using a chessboard pattern
and measures the diameter of circular markers in an image.

Features:
- Automatic scale calculation via Chessboard calibration.
- Interactive threshold adjustment UI.
- Statistical analysis of marker diameters.
- Visualization of results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # File Paths
    "INPUT_IMAGE": "markerdiameter/diameter1.jpg",
    "OUTPUT_IMG": "annotated_result.png",
    "OUTPUT_PLOT": "diameter_statistics.png",

    # Calibration (Chessboard)
    "CHESSBOARD_SIZE": (6, 6),  # Inner corners (rows, columns)
    "SQUARE_SIZE_MM": 3.0,

    # Marker Detection Filters
    "MIN_AREA": 100,
    "MIN_CIRCULARITY": 0.85,  # 1.0 is a perfect circle

    # Calibration Adjustment
    "DIAMETER_OFFSET_MM": 0.0  # Add/Subtract to final measurement
}

# =============================================================================
# UTILITIES
# =============================================================================

def calculate_scale(gray_img, pattern_size, square_mm):
    """Calculates pixels-per-mm scale using a chessboard pattern."""
    print("[INFO] Searching for chessboard corners...")
    found, corners = cv2.findChessboardCorners(gray_img, pattern_size, None)

    if not found:
        print("[ERROR] Chessboard not found. Cannot calculate scale.")
        return None, None

    # Calculate average distance between adjacent corners
    distances = []
    # Horizontal distances
    for r in range(pattern_size[1]):
        for c in range(pattern_size[0] - 1):
            p1 = corners[r * pattern_size[0] + c]
            p2 = corners[r * pattern_size[0] + c + 1]
            distances.append(np.linalg.norm(p1 - p2))
    
    # Vertical distances
    for r in range(pattern_size[1] - 1):
        for c in range(pattern_size[0]):
            p1 = corners[r * pattern_size[0] + c]
            p2 = corners[(r + 1) * pattern_size[0] + c]
            distances.append(np.linalg.norm(p1 - p2))

    avg_px = np.mean(distances)
    scale = avg_px / square_mm  # pixels per mm
    
    print(f"[SUCCESS] Scale calculated: {scale:.2f} px/mm (Avg square: {avg_px:.2f} px)")
    return scale, corners

def select_threshold_interactive(gray_img):
    """Opens a GUI window to manually select the binary threshold."""
    print("\n[INFO] Opening Threshold Adjuster...")
    print("   >>> Instructions: Drag slider to isolate markers. Press ENTER to confirm.")
    
    window_name = "Threshold Selector (Press ENTER to Confirm)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    def nothing(x): pass
    cv2.createTrackbar("Threshold", window_name, 127, 255, nothing)

    final_thresh = -1
    
    while True:
        val = cv2.getTrackbarPos("Threshold", window_name)
        _, binary = cv2.threshold(gray_img, val, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow(window_name, binary)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            final_thresh = val
            break
        
        # Check if window was closed manually
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    
    if final_thresh != -1:
        print(f"[INFO] Threshold selected: {final_thresh}")
    else:
        print("[WARN] Window closed without selection.")
        
    return final_thresh

def measure_markers(gray_img, scale, threshold):
    """Detects contours and calculates diameters in mm."""
    _, thresh = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    diameters = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < CONFIG["MIN_AREA"]: 
            continue
            
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity < CONFIG["MIN_CIRCULARITY"]: 
            continue

        # Measure
        (_, _), radius_px = cv2.minEnclosingCircle(cnt)
        diameter_mm = (radius_px * 2) / scale
        
        # Apply offset compensation if configured
        diameter_mm += CONFIG["DIAMETER_OFFSET_MM"]
        
        valid_contours.append(cnt)
        diameters.append(diameter_mm)

    print(f"[INFO] Detected {len(valid_contours)} valid markers.")
    return valid_contours, diameters

def save_visualizations(original_img, corners, contours, diameters, output_dir):
    """Draws overlays on the image and plots statistical data."""
    # 1. Annotate Image
    annotated = original_img.copy()
    if corners is not None:
        cv2.drawChessboardCorners(annotated, CONFIG["CHESSBOARD_SIZE"], corners, True)
    
    if contours:
        for i, cnt in enumerate(contours):
            cv2.drawContours(annotated, [cnt], -1, (0, 255, 0), 2)
            (x, y), r = cv2.minEnclosingCircle(cnt)
            
            label = f"{diameters[i]:.2f}mm"
            cv2.putText(annotated, label, (int(x)-20, int(y)-int(r)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    img_path = os.path.join(output_dir, CONFIG["OUTPUT_IMG"])
    cv2.imwrite(img_path, annotated)
    print(f"[SUCCESS] Annotated image saved: {img_path}")

    # 2. Plot Statistics
    if not diameters: return

    avg = np.mean(diameters)
    std = np.std(diameters)
    ids = np.arange(1, len(diameters) + 1)
    target = CONFIG["SQUARE_SIZE_MM"] # Assuming markers should match square size roughly, or customize

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(ids, diameters, color='skyblue', edgecolor='black', label='Measured')
    ax.axhline(target, color='red', linestyle='--', label=f'Ref ({target}mm)')
    
    stats_text = (f'Count: {len(diameters)}\n'
                  f'Mean: {avg:.2f} mm\n'
                  f'Std Dev: {std:.2f} mm')
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.8))

    ax.set_title('Marker Diameter Analysis')
    ax.set_xlabel('Marker ID')
    ax.set_ylabel('Diameter (mm)')
    ax.legend()
    
    plot_path = os.path.join(output_dir, CONFIG["OUTPUT_PLOT"])
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"[SUCCESS] Statistics plot saved: {plot_path}")
    # plt.show() # Uncomment if you want to see the plot immediately

# =============================================================================
# MAIN
# =============================================================================

def main():
    img_path = CONFIG["INPUT_IMAGE"]
    if not os.path.exists(img_path):
        print(f"[FATAL] Image not found: {img_path}")
        sys.exit(1)

    # Load Image
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Calibration
    scale, corners = calculate_scale(gray, CONFIG["CHESSBOARD_SIZE"], CONFIG["SQUARE_SIZE_MM"])
    if scale is None: return

    # 2. Threshold Selection
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = select_threshold_interactive(gray_blur)
    if threshold == -1: return

    # 3. Measurement
    contours, diameters = measure_markers(gray_blur, scale, threshold)
    if not diameters:
        print("[WARN] No markers found.")
        return

    # 4. Results
    output_dir = os.path.dirname(img_path) or "."
    save_visualizations(img, corners, contours, diameters, output_dir)

    print("\n--- Summary ---")
    print(f"Mean Diameter: {np.mean(diameters):.3f} mm")
    print(f"Std Deviation: {np.std(diameters):.3f} mm")

if __name__ == "__main__":
    main()