import asyncio
import base64
import cv2
import json
import numpy as np
import time
import websockets
from websockets.exceptions import ConnectionClosed
from ultralytics import YOLO

# Cấu hình
SERVER_IP = "192.168.137.162"  # Thay đổi IP này thành IP của Raspberry Pi
SERVER_PORT = "8000"

# Tải model YOLOv8m
model = YOLO("yolov8m.pt")

async def connect_to_server():
    uri = f"ws://{SERVER_IP}:{SERVER_PORT}"
    last_fps = None
    last_cpu_temp = None
    
    while True:  # Vòng lặp kết nối lại
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as websocket:
                print(f"Đã kết nối tới server tại {uri}")
                
                while True:
                    try:
                        # Nhận dữ liệu từ server với timeout
                        data = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        data = json.loads(data)
                        
                        # Cập nhật giá trị cuối cùng nếu có dữ liệu mới
                        if 'fps' in data and data['fps'] is not None:
                            last_fps = data['fps']
                        if 'cpu_temp' in data and data['cpu_temp'] is not None:
                            last_cpu_temp = data['cpu_temp']
                        
                        # Giải mã ảnh
                        img_bytes = base64.b64decode(data['image'])
                        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                        
                        if frame is None:
                            print("Lỗi: Không thể giải mã ảnh")
                            continue
                        
                        # Bắt đầu thời gian để đo độ trễ suy luận
                        inference_start = time.time()
                        
                        # Thực hiện phát hiện đối tượng với YOLOv8
                        results = model(frame)
                        
                        # Tính thời gian suy luận
                        inference_latency = (time.time() - inference_start) * 1000  # ms
                        
                        # Vẽ hộp giới hạn trên ảnh
                        for result in results:
                            boxes = result.boxes.cpu().numpy()
                            for i, box in enumerate(boxes):
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = float(box.conf[0])
                                cls = int(box.cls[0])
                                cls_name = result.names[cls]
                                
                                # Vẽ hộp giới hạn
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f"{cls_name}: {conf:.2f}", 
                                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.5, (0, 255, 0), 2)
                        
                        # Hiển thị thông tin FPS và nhiệt độ CPU
                        if last_fps is not None:
                            fps_text = f"Camera FPS: {last_fps:.2f}"
                            cv2.putText(frame, fps_text, (10, 30), 
                                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                        if last_cpu_temp is not None:
                            temp_text = f"CPU Temp: {last_cpu_temp}°C"
                            cv2.putText(frame, temp_text, (10, 70), 
                                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Hiển thị độ trễ suy luận
                        latency_text = f"Inference: {inference_latency:.2f}ms"
                        cv2.putText(frame, latency_text, (10, 110), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Hiển thị khung hình
                        cv2.imshow('YOLOv8 Detection Results', frame)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
                            return  # Thoát sạch sẽ
                            
                    except asyncio.TimeoutError:
                        print("Hết thời gian chờ phản hồi từ server")
                        break
                        
        except ConnectionClosed:
            print("Kết nối bị đóng bởi server, đang cố gắng kết nối lại...")
            cv2.destroyAllWindows()
            await asyncio.sleep(2)  # Chờ trước khi kết nối lại
        except Exception as e:
            print(f"Lỗi kết nối: {e}")
            cv2.destroyAllWindows()
            await asyncio.sleep(2)  # Chờ trước khi kết nối lại

if __name__ == "__main__":
    while True:  # Vòng lặp vô hạn để giữ chương trình chạy
        try:
            asyncio.get_event_loop().run_until_complete(connect_to_server())
        except KeyboardInterrupt:
            print("\nĐang thoát chương trình")
            break
