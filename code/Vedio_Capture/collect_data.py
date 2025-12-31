#!/usr/bin/env python3
import cv2
import os
import time
import sys
from datetime import datetime
from rpi_ws281x import PixelStrip, Color

# Configuration
VIDEO_SAVE_DIR = "/home/zfy/EXP/video"
RESOLUTION = (640, 480)
FPS = 15
LED_COUNT = 12
LED_PIN = 18
LED_BRIGHTNESS = 20

def check_sudo():
    """Verify script is run with sudo"""
    if os.geteuid() != 0:
        print("\n[ERROR] This script requires root privileges")
        print("Please run with sudo:\n")
        print("sudo python3 collect_marker_data.py\n")
        sys.exit(1)

def initialize_leds():
    """Initialize LED strip with error handling"""
    try:
        strip = PixelStrip(LED_COUNT, LED_PIN, brightness=LED_BRIGHTNESS)
        strip.begin()
        return strip
    except Exception as e:
        print(f"[LED ERROR] Initialization failed: {str(e)}")
        print("Troubleshooting:")
        print("1. Check physical wiring connections")
        print("2. Verify sufficient power supply (5V 2A+)")
        print("3. Confirm correct GPIO pin configuration")
        sys.exit(1)

def set_all_white(strip):
    """Set all LEDs to white"""
    try:
        for i in range(strip.numPixels()):
            strip.setPixelColor(i, Color(255, 255, 255))
        strip.show()
    except Exception as e:
        print(f"[LED ERROR] Color setting failed: {str(e)}")

def turn_off_leds(strip):
    """Turn off all LEDs"""
    try:
        for i in range(strip.numPixels()):
            strip.setPixelColor(i, Color(0, 0, 0))
        strip.show()
    except:
        pass  # Silent fail during cleanup

def setup_camera():
    """Initialize camera with error handling"""
    try:
        cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
        cap.set(cv2.CAP_PROP_FPS, FPS)
        
        if not cap.isOpened():
            raise RuntimeError("Camera initialization failed")
        
        actual_res = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"[CAMERA] Configured: {RESOLUTION[0]}x{RESOLUTION[1]} @ {FPS}fps")
        print(f"[CAMERA] Actual: {actual_res[0]}x{actual_res[1]} @ {actual_fps:.1f}fps")
        
        return cap, actual_res
        
    except Exception as e:
        print(f"[CAMERA ERROR] {str(e)}")
        print("Troubleshooting:")
        print("1. Run: v4l2-ctl --list-devices")
        print("2. Test with: ffplay -f v4l2 /dev/video0")
        sys.exit(1)

def record_session():
    """Main recording function"""
    check_sudo()  # Enforce root privileges
    
    # Initialize hardware
    led_strip = initialize_leds()
    set_all_white(led_strip)
    camera, resolution = setup_camera()
    
    # ========================== MODIFICATION START ==========================
    # Discard first 100 frames to allow camera to stabilize
    print("[STATUS] Discarding first 100 frames for camera warm-up...")
    for _ in range(100):
        camera.read()
    print("[STATUS] Camera stabilized. Starting recording...")
    # ========================== MODIFICATION END ============================

    # Create output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(VIDEO_SAVE_DIR, f"marker_data_{timestamp}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(filename, fourcc, FPS, resolution)

    print(f"\n[STATUS] Recording to: {filename}")
    print("Press CTRL+C to stop\n")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("[WARNING] Frame capture failed!")
                break

            out.write(frame)
            frame_count += 1

            # Print status every 5 seconds
            if time.time() - start_time >= 5 and frame_count % 5 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                print(f"[STATUS] Frames: {frame_count} | FPS: {current_fps:.1f}")

    except KeyboardInterrupt:
        print("\n[STATUS] Stopping recording...")

    finally:
        # Calculate metrics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0

        # Release resources
        camera.release()
        out.release()
        turn_off_leds(led_strip)
        
        print(f"理论应保存帧数: {int(total_time * FPS)}")
        print(f"实际保存帧数: {frame_count}")

        # Summary report
        print("\n[SUMMARY]")
        print(f"Duration: {total_time:.1f}s")
        print(f"Frames captured: {frame_count}")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"File saved: {filename}")
        print(f"File size: {os.path.getsize(filename)/1024:.1f} KB")

if __name__ == "__main__":
    print("="*50)
    print("MARKER DATA COLLECTION SYSTEM")
    print("="*50)
    record_session()