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

# Cấu hình camera với độ phân giải thấp hơn và tối ưu hóa
width, height = 640, 480  # Giảm độ phân giải để tăng tốc độ xử lý
picam2 = Picamera2()
capture_config = picam2.create_preview_configuration(
    main={"format": "YUV420", "size": (width, height)},
    controls={"FrameRate": 15.0},
    buffer_count=4  # Tăng số lượng buffer để tránh bottleneck
)
picam2.configure(capture_config)

# Cấu hình nén JPEG
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]  # Giảm chất lượng JPEG để giảm kích thước dữ liệu

# Queue để chia sẻ dữ liệu giữa các luồng (giữ nhỏ để giảm độ trễ)
frame_queue = queue.Queue(maxsize=2)
data_queue = queue.Queue(maxsize=2)

# Biến kiểm soát luồng
running = True

def get_cpu_temp():
    try:
        temp = os.popen("vcgencmd measure_temp").readline().replace("temp=", "").replace("'C\n", "")
        return float(temp)
    except:
        return 0.0

def capture_frames():
    """Luồng riêng biệt để thu thập khung hình từ camera"""
    frame_count = 0
    start_time = time.time()
    fps = 0
    cpu_temp = 0
    fps_update_interval = 30  # Cập nhật FPS sau mỗi 30 khung hình
    
    while running:
        try:
            # Bỏ qua nếu queue đã đầy để tránh tích tụ
            if frame_queue.full():
                time.sleep(0.001)
                continue
                
            # Chụp ảnh
            frame = picam2.capture_array("main")
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV420p2RGB)
            
            # Xoay ảnh 90 độ ngược chiều kim đồng hồ
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Tính FPS và nhiệt độ CPU (giảm tần suất đo để tiết kiệm CPU)
            frame_count += 1
            if frame_count % fps_update_interval == 0:
                end_time = time.time()
                fps = fps_update_interval / (end_time - start_time)
                cpu_temp = get_cpu_temp()
                start_time = time.time()
            
            # Đưa frame vào queue để xử lý
            frame_queue.put((frame, fps, cpu_temp), block=False)
                
        except queue.Full:
            # Bỏ qua nếu queue đầy
            pass
        except Exception as e:
            print(f"Lỗi khi chụp khung hình: {e}")
            time.sleep(0.1)

def process_frames():
    """Luồng riêng biệt để xử lý khung hình và mã hóa"""
    while running:
        try:
            if frame_queue.empty():
                time.sleep(0.01)
                continue
                
            frame, fps, cpu_temp = frame_queue.get(block=False)
            
            # Giảm kích thước ảnh nếu cần
            # frame = cv2.resize(frame, (320, 240))
            
            # Mã hóa ảnh với chất lượng thấp hơn
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            img_str = base64.b64encode(buffer).decode('utf-8')
            
            # Giảm dữ liệu gửi đi
            data = {
                "image": img_str,
                # Chỉ gửi FPS và nhiệt độ khi có cập nhật mới
                "fps": round(fps, 1) if fps > 0 else None,
                "cpu_temp": round(cpu_temp, 1) if cpu_temp > 0 else None
            }
            
            # Đưa dữ liệu đã xử lý vào queue để gửi
            if not data_queue.full():
                data_queue.put(data, block=False)
            
            # Đánh dấu là đã xử lý xong
            frame_queue.task_done()
            
        except queue.Empty:
            # Bỏ qua nếu không có frame
            pass
        except Exception as e:
            print(f"Lỗi khi xử lý khung hình: {e}")
            time.sleep(0.01)

async def process_client(websocket):
    global running
    running = True
    
    picam2.start()
    
    # Khởi động các luồng xử lý với độ ưu tiên cao
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    process_thread = threading.Thread(target=process_frames, daemon=True)
    
    capture_thread.start()
    process_thread.start()
    
    try:
        while running:
            # Chờ và gửi dữ liệu
            try:
                if data_queue.empty():
                    await asyncio.sleep(0.01)
                    continue
                    
                data = data_queue.get(block=False)
                await websocket.send(json.dumps(data))
                data_queue.task_done()
                
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                print(f"Lỗi gửi dữ liệu: {e}")
                await asyncio.sleep(0.1)
            
    except Exception as e:
        print(f"Lỗi kết nối: {e}")
    finally:
        running = False
        picam2.stop()
        # Xóa hết queue để tránh memory leak
        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
                frame_queue.task_done()
            except:
                pass
        while not data_queue.empty():
            try:
                data_queue.get_nowait()
                data_queue.task_done()
            except:
                pass
        print("Đang chờ kết nối mới...")

async def main():
    server = await websockets.serve(
        process_client,
        "0.0.0.0",
        8000,
        ping_interval=30,
        ping_timeout=30,
        max_size=None,  # Không giới hạn kích thước tin nhắn
        compression=None  # Tắt nén websocket vì chúng ta đã nén JPEG
    )
    print("Server camera đã khởi động tại ws://0.0.0.0:8000")
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        running = False
        print("\nServer đang tắt")
