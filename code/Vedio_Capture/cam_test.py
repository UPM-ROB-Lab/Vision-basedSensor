#!/usr/bin/env python3
import cv2
import time
import threading
import os
import sys

# 导入 rpi_ws281x 库
try:
    from rpi_ws281x import PixelStrip, Color
except ImportError:
    # 在非树莓派环境下运行，提供模拟类以防止报错
    class PixelStrip:
        def __init__(self, *args, **kwargs):
            print("[WARN] rpi_ws281x not found. LED control will be simulated.")
        def begin(self): pass
        def setPixelColor(self, i, color): pass
        def show(self): pass
        def numPixels(self): return 12
    class Color:
        def __init__(self, r, g, b): pass

from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import numpy as np

# 基础配置
CAMERA_INDEX = 0      # 尝试不同的索引值(0,1,2等)直到找到你的摄像头
WIDTH = 640           # 视频宽度
HEIGHT = 480          # 视频高度
FPS = 12              # 目标帧率
PORT = 8081           # 服务端口
SKIP_FRAMES = 1       # 跳帧数(降低CPU使用率)
LED_COUNT = 12        # LED灯数量
LED_PIN = 18          # LED灯连接的GPIO引脚
LED_BRIGHTNESS = 20   # LED灯亮度 (0-255)

# --- LED控制函数 ---
def initialize_leds():
    """初始化LED灯带"""
    try:
        strip = PixelStrip(LED_COUNT, LED_PIN, brightness=LED_BRIGHTNESS)
        strip.begin()
        return strip
    except Exception as e:
        print(f"[LED ERROR] 初始化失败: {str(e)}")
        return None

def set_all_white(strip):
    """将所有LED设置为白色"""
    if not strip: return
    try:
        for i in range(strip.numPixels()):
            strip.setPixelColor(i, Color(255, 255, 255))
        strip.show()
    except:
        pass

def turn_off_leds(strip):
    """关闭所有LED"""
    if not strip: return
    try:
        for i in range(strip.numPixels()):
            strip.setPixelColor(i, Color(0, 0, 0))
        strip.show()
    except:
        pass

# --- 摄像头和流媒体部分 ---
class StreamingServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class CameraHandler:
    def __init__(self, led_strip):
        self.led_strip = led_strip
        self.cap = None
        self.frame = None
        self.running = True
        self._init_camera()

    def _init_camera(self):
        """尝试初始化摄像头"""
        # 在初始化摄像头前打开LED灯
        set_all_white(self.led_strip)
        time.sleep(1) # 等待灯光稳定
        
        for i in range(3):
            self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
                self.cap.set(cv2.CAP_PROP_FPS, FPS)
                print(f"Camera opened at {WIDTH}x{HEIGHT}@{FPS}fps")
                return
            time.sleep(1)
            
        print(f"[WARN] 无法打开摄像头 /dev/video{CAMERA_INDEX}，将显示测试画面")
        self.cap = None

    def capture_loop(self):
        """持续捕获视频帧"""
        frame_count = 0
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
            else:
                frame = self._generate_test_frame()
                time.sleep(1.0 / FPS)
                frame_count += 1
                continue

            frame_count += 1
            if frame_count % (SKIP_FRAMES + 1) != 0:
                continue

            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            self.frame = jpeg.tobytes()

    def _generate_test_frame(self):
        """生成测试帧(当摄像头不可用时)"""
        img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        cv2.putText(img, "NO CAMERA", (50, HEIGHT//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
        return img

    def get_frame(self):
        """获取当前帧"""
        if self.frame is not None:
            return self.frame
        _, jpeg = cv2.imencode('.jpg', self._generate_test_frame())
        return jpeg.tobytes()

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = f'''
                <html><body>
                <img src="/stream" width="{WIDTH}">
                <p>Camera Stream {WIDTH}x{HEIGHT}@{FPS}fps</p>
                </body></html>
            '''
            self.wfile.write(html.encode())
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    frame = camera_handler.get_frame()
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    time.sleep(1.0 / max(1, FPS))
            except (ConnectionError, BrokenPipeError):
                pass
        else:
            self.send_error(404)

def run_server():
    global camera_handler
    led_strip = initialize_leds()
    
    # 强制以root权限运行
    if os.geteuid() != 0:
        print("\n[ERROR] 该脚本需要root权限。请使用 sudo 运行:")
        print("sudo python3 your_script_name.py")
        sys.exit(1)

    camera_handler = CameraHandler(led_strip)
    threading.Thread(target=camera_handler.capture_loop, daemon=True).start()
    
    print(f"\n服务已启动: http://0.0.0.0:{PORT}")
    print(f"在您的Windows电脑上，请访问: http://<树莓派IP地址>:{PORT}")
    try:
        StreamingServer(('0.0.0.0', PORT), RequestHandler).serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭服务...")
    finally:
        camera_handler.running = False
        if camera_handler.cap:
            camera_handler.cap.release()
        turn_off_leds(led_strip)

if __name__ == '__main__':
    run_server()