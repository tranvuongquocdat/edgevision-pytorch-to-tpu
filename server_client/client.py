import asyncio
import base64
import cv2
import json
import numpy as np
import time
import websockets
import threading
import queue
from websockets.exceptions import ConnectionClosed
from ultralytics import YOLO

# Cấu hình
SERVER_IP = "192.168.137.81"  # Thay đổi IP này thành IP của Raspberry Pi
SERVER_PORT = "8000"

# Tải model YOLOv8m và thiết lập tham số
model = YOLO("yolov8m.pt")
model.conf = 0.25  # Giảm ngưỡng tin cậy để phát hiện tốt hơn với ảnh chất lượng thấp
model.iou = 0.45  # Điều chỉnh IoU

# Queue để xử lý khung hình trong luồng riêng biệt
frame_queue = queue.Queue(maxsize=3)
result_queue = queue.Queue(maxsize=3)

# Biến toàn cục để chia sẻ giữa các luồng
running = True
last_fps = None
last_cpu_temp = None

def process_frame_thread():
    """Luồng xử lý YOLOv8 riêng biệt để tránh block UI"""
    global running
    
    while running:
        try:
            if frame_queue.empty():
                time.sleep(0.01)
                continue
                
            frame = frame_queue.get(block=False)
            
            # Thực hiện phát hiện đối tượng với YOLOv8
            inference_start = time.time()
            results = model(frame, verbose=False)
            inference_latency = (time.time() - inference_start) * 1000  # ms
            
            # Đưa kết quả vào queue
            if not result_queue.full():
                result_queue.put((frame, results, inference_latency))
                
            frame_queue.task_done()
                
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Lỗi trong luồng xử lý: {e}")
            time.sleep(0.01)

def render_results_thread():
    """Luồng hiển thị kết quả riêng biệt"""
    global running, last_fps, last_cpu_temp
    
    while running:
        try:
            if result_queue.empty():
                # Hiển thị thông báo đang xử lý nếu không có kết quả
                time.sleep(0.01)
                continue
                
            frame, results, inference_latency = result_queue.get(block=False)
            
            # Vẽ hộp giới hạn trên ảnh
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    cls_name = result.names[cls]
                    
                    # Vẽ hộp giới hạn với màu dựa trên lớp
                    color = (0, 255, 0)  # Mặc định màu xanh lá
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{cls_name}: {conf:.2f}", 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, color, 2)
            
            # Hiển thị thông tin FPS và nhiệt độ CPU
            if last_fps is not None:
                fps_text = f"Camera FPS: {last_fps:.1f}"
                cv2.putText(frame, fps_text, (10, 30), 
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            if last_cpu_temp is not None:
                temp_text = f"CPU Temp: {last_cpu_temp}°C"
                cv2.putText(frame, temp_text, (10, 70), 
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Hiển thị độ trễ suy luận
            latency_text = f"Inference: {inference_latency:.1f}ms"
            cv2.putText(frame, latency_text, (10, 110), 
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Hiển thị khung hình
            cv2.imshow('YOLOv8 Detection Results', frame)
            
            # Chờ phím trong thời gian ngắn để không block UI
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                running = False
                cv2.destroyAllWindows()
                print("Đã nhấn phím q, đang thoát chương trình...")
                break
                
            result_queue.task_done()
                
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Lỗi trong luồng hiển thị: {e}")
            time.sleep(0.01)

async def connect_to_server():
    global running, last_fps, last_cpu_temp
    uri = f"ws://{SERVER_IP}:{SERVER_PORT}"
    
    # Khởi động các luồng xử lý
    process_thread = threading.Thread(target=process_frame_thread, daemon=True)
    render_thread = threading.Thread(target=render_results_thread, daemon=True)
    
    process_thread.start()
    render_thread.start()
    
    while running:  # Vòng lặp kết nối lại
        try:
            async with websockets.connect(uri, ping_interval=30, ping_timeout=30) as websocket:
                print(f"Đã kết nối tới server tại {uri}")
                
                while running:
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
                        
                        # Tăng độ sáng và độ tương phản nếu cần
                        # alpha = 1.1  # Tương phản (1.0-3.0)
                        # beta = 10    # Độ sáng (0-100)
                        # frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
                        
                        # Đưa frame vào queue để xử lý trong luồng riêng
                        if not frame_queue.full():
                            frame_queue.put(frame)
                            
                    except asyncio.TimeoutError:
                        print("Hết thời gian chờ phản hồi từ server")
                        break
                    except Exception as e:
                        print(f"Lỗi khi nhận dữ liệu: {e}")
                        await asyncio.sleep(0.1)
                        
        except ConnectionClosed:
            print("Kết nối bị đóng bởi server, đang cố gắng kết nối lại...")
            await asyncio.sleep(2)  # Chờ trước khi kết nối lại
        except Exception as e:
            print(f"Lỗi kết nối: {e}")
            await asyncio.sleep(2)  # Chờ trước khi kết nối lại
            
        # Kiểm tra trạng thái running
        if not running:
            break

async def shutdown():
    """Hàm đóng các tài nguyên khi thoát"""
    global running
    running = False
    cv2.destroyAllWindows()
    
    # Xóa hết queue để tránh memory leak
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
            frame_queue.task_done()
        except:
            pass
            
    while not result_queue.empty():
        try:
            result_queue.get_nowait()
            result_queue.task_done()
        except:
            pass

if __name__ == "__main__":
    try:
        # Đặt chế độ hiệu suất cao cho OpenCV
        cv2.setNumThreads(4)  # Sử dụng 4 luồng cho xử lý OpenCV
        
        # Bắt đầu chương trình
        asyncio.run(connect_to_server())
    except KeyboardInterrupt:
        print("\nĐang thoát chương trình")
    finally:
        # Đảm bảo đóng tất cả tài nguyên
        asyncio.run(shutdown())
