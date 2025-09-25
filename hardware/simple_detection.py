import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
from mlx90640_camera import MLX90640Camera
import os

def simple_detection():
    """Simple detection without real-time display - good for testing"""
    
    # Fix for PyTorch 2.8.0+ weights_only issue
    original_load = torch.load
    def safe_load(file, **kwargs):
        return original_load(file, weights_only=False, **kwargs)
    torch.load = safe_load
    
    # Try to find the latest trained model
    import glob
    model_files = glob.glob("runs/detect/*/weights/best.pt")
    
    if model_files:
        latest_model = max(model_files, key=os.path.getctime)
        print(f"Using trained model: {latest_model}")
        model = YOLO(latest_model)
    else:
        print("No trained model found. Using base model.")
        model = YOLO('yolov8n.yaml')
    
    # Initialize camera
    try:
        camera = MLX90640Camera()
        print("MLX90640 camera initialized successfully")
    except Exception as e:
        print(f"Failed to initialize MLX90640 camera: {e}")
        print("Please check your camera connection and try again.")
        return
    
    print("Press 'q' to quit, 's' to save frame")
    
    try:
        while True:
            # Capture thermal frame
            thermal_array = camera.capture_frame()
            
            if thermal_array is not None:
                # Convert to visual
                visual_frame = camera.thermal_to_visual(thermal_array)
                
                # Resize for YOLO
                frame = camera.resize_for_yolo(visual_frame)
                
                # Get temperature stats
                temp_stats = camera.get_temperature_stats(thermal_array)
                
                # Run detection
                results = model(frame, conf=0.25)
                
                # Process results
                if len(results) > 0:
                    result = results[0]
                    
                    # Draw detections
                    annotated_frame = result.plot()
                    
                    # Add temperature info
                    if temp_stats:
                        cv2.putText(annotated_frame, 
                                   f"Center Temp: {temp_stats['center_temp']:.1f}°C", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Display frame
                    cv2.imshow('MLX90640 Detection', annotated_frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"hardware/captured_frames/thermal_frame_{timestamp}.jpg"
                        os.makedirs("hardware/captured_frames", exist_ok=True)
                        cv2.imwrite(filename, annotated_frame)
                        print(f"Frame saved: {filename}")
                
                else:
                    # No detections
                    cv2.imshow('MLX90640 Detection', frame)
                    cv2.waitKey(1)
            
            # Small delay
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        camera.close()
        cv2.destroyAllWindows()
        print("Detection stopped.")

if __name__ == "__main__":
    simple_detection()



