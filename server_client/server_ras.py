#!/usr/bin/python3

import asyncio
import base64
import cv2
import json
import os
import time
import websockets
import threading
import queue
from pathlib import Path
from picamera2 import Picamera2
import numpy as np

# Cấu hình camera
width, height = 640, 480
picam2 = Picamera2()
capture_config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (width, height)}
)
picam2.configure(capture_config)

# Queue để chia sẻ dữ liệu giữa các luồng
frame_queue = queue.Queue(maxsize=10)
data_queue = queue.Queue(maxsize=10)

def get_cpu_temp():
    temp = os.popen("vcgencmd measure_temp").readline().replace("temp=", "").replace("'C\n", "")
    return float(temp)

def capture_frames():
    """Luồng riêng biệt để thu thập khung hình từ camera"""
    frame_count = 0
    start_time = time.time()
    fps = 0
    cpu_temp = 0
    
    while True:
        try:
            # Chụp ảnh
            pil_image = picam2.capture_image()
            # Chuyển đổi ảnh PIL thành numpy array cho OpenCV
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Xoay ảnh 90 độ ngược chiều kim đồng hồ
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Tính FPS và nhiệt độ CPU
            frame_count += 1
            if frame_count % 10 == 0:
                end_time = time.time()
                fps = 10 / (end_time - start_time)
                cpu_temp = get_cpu_temp()
                start_time = time.time()
            
            # Đưa frame vào queue để xử lý
            if not frame_queue.full():
                frame_queue.put((frame, fps, cpu_temp))
                
        except Exception as e:
            print(f"Lỗi khi chụp khung hình: {e}")
            time.sleep(0.1)

def process_frames():
    """Luồng riêng biệt để xử lý khung hình và mã hóa"""
    while True:
        try:
            if not frame_queue.empty():
                frame, fps, cpu_temp = frame_queue.get()
                
                # Mã hóa ảnh
                _, buffer = cv2.imencode('.jpg', frame)
                img_str = base64.b64encode(buffer).decode('utf-8')
                
                # Đưa dữ liệu đã xử lý vào queue để gửi
                if not data_queue.full():
                    data_queue.put({
                        "image": img_str,
                        "fps": fps,
                        "cpu_temp": cpu_temp
                    })
                
                # Đánh dấu là đã xử lý xong
                frame_queue.task_done()
        except Exception as e:
            print(f"Lỗi khi xử lý khung hình: {e}")
            time.sleep(0.1)

async def process_client(websocket):
    picam2.start()
    
    # Khởi động các luồng xử lý
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    process_thread = threading.Thread(target=process_frames, daemon=True)
    
    capture_thread.start()
    process_thread.start()
    
    try:
        while True:
            # Chờ và gửi dữ liệu
            try:
                if not data_queue.empty():
                    data = data_queue.get()
                    await websocket.send(json.dumps(data))
                    data_queue.task_done()
                else:
                    # Không có dữ liệu, chờ một chút
                    await asyncio.sleep(0.01)
            except websockets.exceptions.ConnectionClosed:
                break
            
    except Exception as e:
        print(f"Lỗi kết nối: {e}")
    finally:
        picam2.stop()
        print("Đang chờ kết nối mới...")

async def main():
    server = await websockets.serve(
        process_client,
        "0.0.0.0",
        8000,
        ping_interval=20,
        ping_timeout=20
    )
    print("Server camera đã khởi động tại ws://0.0.0.0:8000")
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer đang tắt")
