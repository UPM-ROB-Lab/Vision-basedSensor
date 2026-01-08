#!/usr/bin/env python3
import cv2
import time
import threading
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import numpy as np

# Import rpi_ws281x library with fallback
try:
    from rpi_ws281x import PixelStrip, Color
except ImportError:
    # Fallback for non-Raspberry Pi environments
    class PixelStrip:
        def __init__(self, *args, **kwargs):
            print("[WARNING] rpi_ws281x not found. LED control will be simulated.")
        def begin(self): pass
        def setPixelColor(self, i, color): pass
        def show(self): pass
        def numPixels(self): return 12
    class Color:
        def __init__(self, r, g, b): pass

# Configuration constants
CONFIG = {
    'CAMERA_INDEX': 0,      # Try different indices (0,1,2) to find your camera
    'WIDTH': 640,           # Video width
    'HEIGHT': 480,          # Video height
    'FPS': 12,              # Target frame rate
    'PORT': 8081,           # Server port
    'SKIP_FRAMES': 1,       # Frame skip count (reduces CPU usage)
    'LED_COUNT': 12,        # Number of LED lights
    'LED_PIN': 18,          # GPIO pin for LEDs
    'LED_BRIGHTNESS': 20    # LED brightness (0-255)
}

class LEDController:
    """Handles LED strip initialization and control"""
    
    def __init__(self):
        self.strip = self._initialize_strip()
        
    def _initialize_strip(self):
        """Initialize the LED strip"""
        try:
            strip = PixelStrip(
                CONFIG['LED_COUNT'],
                CONFIG['LED_PIN'],
                brightness=CONFIG['LED_BRIGHTNESS']
            )
            strip.begin()
            return strip
        except Exception as e:
            print(f"[LED ERROR] Initialization failed: {str(e)}")
            return None
    
    def set_all_white(self):
        """Set all LEDs to white"""
        if not self.strip:
            return
        try:
            for i in range(self.strip.numPixels()):
                self.strip.setPixelColor(i, Color(255, 255, 255))
            self.strip.show()
        except Exception as e:
            print(f"[LED WARNING] Failed to set white: {str(e)}")
    
    def turn_off(self):
        """Turn off all LEDs"""
        if not self.strip:
            return
        try:
            for i in range(self.strip.numPixels()):
                self.strip.setPixelColor(i, Color(0, 0, 0))
            self.strip.show()
        except Exception as e:
            print(f"[LED WARNING] Failed to turn off: {str(e)}")

class CameraHandler:
    """Handles camera initialization, frame capture, and streaming"""
    
    def __init__(self, led_controller):
        self.led_controller = led_controller
        self.cap = None
        self.frame = None
        self.running = True
        self._init_camera()
    
    def _init_camera(self):
        """Initialize the camera device"""
        # Turn on LEDs before camera initialization
        self.led_controller.set_all_white()
        time.sleep(1)  # Wait for light stabilization
        
        for _ in range(3):  # Try 3 times
            self.cap = cv2.VideoCapture(CONFIG['CAMERA_INDEX'], cv2.CAP_V4L2)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['WIDTH'])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['HEIGHT'])
                self.cap.set(cv2.CAP_PROP_FPS, CONFIG['FPS'])
                print(f"Camera initialized at {CONFIG['WIDTH']}x{CONFIG['HEIGHT']} @ {CONFIG['FPS']}fps")
                return
            time.sleep(1)
        
        print(f"[WARNING] Failed to open camera /dev/video{CONFIG['CAMERA_INDEX']}, using test pattern")
        self.cap = None
    
    def capture_loop(self):
        """Main frame capture loop"""
        frame_count = 0
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
            else:
                frame = self._generate_test_frame()
                time.sleep(1.0 / CONFIG['FPS'])
                frame_count += 1
                continue
            
            frame_count += 1
            if frame_count % (CONFIG['SKIP_FRAMES'] + 1) != 0:
                continue
                
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            self.frame = jpeg.tobytes()
    
    def _generate_test_frame(self):
        """Generate a test frame when camera is unavailable"""
        img = np.zeros((CONFIG['HEIGHT'], CONFIG['WIDTH'], 3), np.uint8)
        cv2.putText(
            img, "NO CAMERA", 
            (50, CONFIG['HEIGHT']//2), 
            cv2.FONT_HERSHEY_SIMPLEX, 2, 
            (255, 255, 255), 3
        )
        return img
    
    def get_frame(self):
        """Get the current frame"""
        return self.frame if self.frame is not None else self._generate_encoded_test_frame()
    
    def _generate_encoded_test_frame(self):
        """Generate and encode a test frame"""
        _, jpeg = cv2.imencode('.jpg', self._generate_test_frame())
        return jpeg.tobytes()

class StreamingRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for video streaming"""
    
    def do_GET(self):
        if self.path == '/':
            self._serve_home_page()
        elif self.path == '/stream':
            self._serve_video_stream()
        else:
            self.send_error(404)
    
    def _serve_home_page(self):
        """Serve the HTML page with video stream"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        html = f'''
            <html><body>
            <img src="/stream" width="{CONFIG['WIDTH']}">
            <p>Camera Stream {CONFIG['WIDTH']}x{CONFIG['HEIGHT']} @ {CONFIG['FPS']}fps</p>
            </body></html>
        '''
        self.wfile.write(html.encode())
    
    def _serve_video_stream(self):
        """Serve the MJPEG video stream"""
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        try:
            while True:
                frame = camera_handler.get_frame()
                self.wfile.write(
                    b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + 
                    frame + b'\r\n'
                )
                time.sleep(1.0 / max(1, CONFIG['FPS']))
        except (ConnectionError, BrokenPipeError):
            pass

class StreamingServer(ThreadingMixIn, HTTPServer):
    """Threaded HTTP server for video streaming"""
    daemon_threads = True

def run_server():
    """Main function to run the streaming server"""
    global camera_handler
    
    # Require root privileges
    if os.geteuid() != 0:
        print("\n[ERROR] This script requires root privileges. Please run with:")
        print("sudo python3 your_script_name.py")
        sys.exit(1)
    
    # Initialize components
    led_controller = LEDController()
    camera_handler = CameraHandler(led_controller)
    
    # Start capture thread
    threading.Thread(target=camera_handler.capture_loop, daemon=True).start()
    
    # Start server
    print(f"\nServer started: http://0.0.0.0:{CONFIG['PORT']}")
    print(f"On your Windows machine, access: http://<RaspberryPi_IP>:{CONFIG['PORT']}")
    
    try:
        StreamingServer(('0.0.0.0', CONFIG['PORT']), StreamingRequestHandler).serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        # Cleanup
        camera_handler.running = False
        if camera_handler.cap:
            camera_handler.cap.release()
        led_controller.turn_off()

if __name__ == '__main__':
    run_server()