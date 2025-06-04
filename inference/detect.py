from edgetpumodel import EdgeTPUModel
from utils import get_image_tensor
import picamera2
import numpy as np
import time
import os
import argparse
import signal
import sys

def signal_handler(sig, frame):
    print('Interrupt received, shutting down...')
    camera.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Add argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="models/yolo11n_320.tflite",
                    help='Path to the model file')
parser.add_argument('--names_path', type=str, default="models/label_yolo11.yaml",
                    help='Path to the labels file')
args = parser.parse_args()

print("Initializing camera...")
# Initialize the camera
camera = picamera2.Picamera2()
camera_config = camera.create_preview_configuration(main={"format": "RGB888","size": (640, 480)})
camera.configure(camera_config)
camera.start()
print("Camera initialized successfully")

print("Loading EdgeTPU model...")
# Use command line arguments for model paths
try:
    model = EdgeTPUModel(args.model_path, args.names_path, conf_thresh=0.5, iou_thresh=0.25)
    input_shape = model.get_image_size()
    print(f"Model loaded successfully, input shape: {input_shape}")
except Exception as e:
    print(f"Error loading model: {e}")
    camera.stop()
    sys.exit(1)

# Variables to calculate FPS
frame_count = 0
start_time = time.time()

print("Starting inference loop...")
try:
    while True:
        print(f"Frame {frame_count + 1}: Capturing image...")
        # Capture image
        full_image = camera.capture_array()
        print(f"Image captured, shape: {full_image.shape}")
        
        print("Preprocessing image...")
        # Resize and preprocess image
        full_image, net_image, pad = get_image_tensor(full_image, input_shape[0])
        print(f"Image preprocessed, net_image shape: {net_image.shape}")
        
        print("Running inference...")
        # Predict
        pred = model.forward(net_image)
        print("Inference completed")
        
        print("Processing predictions...")        
        det = model.process_predictions(pred[0], full_image, pad)        
        print(f"Results: {det}")
        
        # Count frames and calculate FPS
        frame_count += 1
        if frame_count % 20 == 0:
            end_time = time.time()
            fps = 20 / (end_time - start_time)
            cpu_temp = os.popen("vcgencmd measure_temp").readline()
            print(f"FPS: {fps:.2f}, CPU Temp: {cpu_temp}")
            start_time = time.time()
            
        # Add small delay to prevent overwhelming the system
        time.sleep(0.01)

except KeyboardInterrupt:
    print("Stopping...")
except Exception as e:
    print(f"Error during inference: {e}")
finally:
    print("Cleaning up...")
    camera.stop()
    print("Camera stopped")