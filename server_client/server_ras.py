#!/usr/bin/python3

import asyncio
import base64
import cv2
import json
import os
import time
import websockets
from pathlib import Path
from picamera2 import Picamera2

# Cấu hình camera
width, height = 640, 480
picam2 = Picamera2()
capture_config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (width, height)}
)
picam2.configure(capture_config)

def get_cpu_temp():
    temp = os.popen("vcgencmd measure_temp").readline().replace("temp=", "").replace("'C\n", "")
    return float(temp)

async def process_client(websocket):
    picam2.start()
    
    frame_count = 0
    start_time = time.time()
    fps = 0
    cpu_temp = 0
    
    try:
        while True:
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
            
            # Mã hóa ảnh
            _, buffer = cv2.imencode('.jpg', frame)
            img_str = base64.b64encode(buffer).decode('utf-8')
            
            # Gửi dữ liệu
            try:
                await websocket.send(json.dumps({
                    "image": img_str,
                    "fps": fps,
                    "cpu_temp": cpu_temp
                }))
            except websockets.exceptions.ConnectionClosed:
                break
            
            # Độ trễ nhỏ để tránh quá tải mạng
            await asyncio.sleep(0.01)
            
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
    import numpy as np
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer đang tắt")
