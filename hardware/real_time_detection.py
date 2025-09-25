import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
from mlx90640_camera import MLX90640Camera
import os

class RealTimeSolarDefectDetector:
    """Real-time solar defect detection using MLX90640 IR camera and YOLO"""
    
    def __init__(self, model_path=None, confidence=0.25):
        """Initialize the real-time detector"""
        
        # Fix for PyTorch 2.8.0+ weights_only issue
        original_load = torch.load
        def safe_load(file, **kwargs):
            return original_load(file, weights_only=False, **kwargs)
        torch.load = safe_load
        
        # Load YOLO model
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"Loaded trained model: {model_path}")
        else:
            # Try to find the latest trained model
            import glob
            model_files = glob.glob("runs/detect/*/weights/best.pt")
            if model_files:
                latest_model = max(model_files, key=os.path.getctime)
                self.model = YOLO(latest_model)
                print(f"Loaded latest model: {latest_model}")
            else:
                print("No trained model found. Using base model.")
                self.model = YOLO('yolov8n.yaml')
        
        # Initialize camera
        try:
            self.camera = MLX90640Camera()
            print("MLX90640 camera initialized successfully")
        except Exception as e:
            print(f"Failed to initialize MLX90640 camera: {e}")
            print("Using webcam as fallback...")
            self.camera = None
            self.webcam = cv2.VideoCapture(0)
        
        # Detection settings
        self.confidence = confidence
        self.running = False
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.fps = 0
        
    def process_frame(self, frame):
        """Process a single frame with YOLO detection"""
        if frame is None:
            return None
            
        # Run YOLO detection
        results = self.model(frame, conf=self.confidence)
        
        # Get the first result
        if len(results) > 0:
            result = results[0]
            
            # Draw detection boxes
            annotated_frame = result.plot()
            
            # Add detection info
            if result.boxes is not None and len(result.boxes) > 0:
                detections = []
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]
                    conf = float(box.conf[0])
                    detections.append(f"{class_name}: {conf:.2f}")
                
                # Display detections
                y_offset = 30
                for detection in detections:
                    cv2.putText(annotated_frame, detection, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_offset += 25
        else:
            annotated_frame = frame
            
        return annotated_frame
    
    def add_overlay(self, frame, thermal_stats=None):
        """Add information overlay to the frame"""
        if frame is None:
            return frame
            
        # Add FPS counter
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add thermal information if available
        if thermal_stats:
            temp_info = f"Temp: {thermal_stats['center_temp']:.1f}°C"
            cv2.putText(frame, temp_info, (10, frame.shape[0] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add instructions
        cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        """Main real-time detection loop"""
        print("Starting real-time detection...")
        print("Press 'q' to quit, 's' to save current frame")
        
        self.running = True
        
        while self.running:
            start_time = time.time()
            
            # Capture frame
            if self.camera:
                # Use MLX90640 camera
                thermal_array = self.camera.capture_frame()
                if thermal_array is not None:
                    # Convert thermal to visual
                    visual_frame = self.camera.thermal_to_visual(thermal_array)
                    # Resize for YOLO
                    frame = self.camera.resize_for_yolo(visual_frame)
                    thermal_stats = self.camera.get_temperature_stats(thermal_array)
                else:
                    frame = None
                    thermal_stats = None
            else:
                # Use webcam as fallback
                ret, frame = self.webcam.read()
                if not ret:
                    frame = None
                thermal_stats = None
            
            if frame is not None:
                # Process frame with YOLO
                processed_frame = self.process_frame(frame)
                
                # Add overlay
                final_frame = self.add_overlay(processed_frame, thermal_stats)
                
                # Display frame
                cv2.imshow('Real-Time Solar Defect Detection', final_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('s'):
                    self.save_frame(final_frame)
                
                # Calculate FPS
                self.fps_counter += 1
                if time.time() - self.fps_start_time >= 1.0:
                    self.fps = self.fps_counter
                    self.fps_counter = 0
                    self.fps_start_time = time.time()
            
            # Control frame rate
            elapsed = time.time() - start_time
            if elapsed < 0.033:  # Target ~30 FPS
                time.sleep(0.033 - elapsed)
        
        self.cleanup()
    
    def save_frame(self, frame):
        """Save current frame"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"hardware/captured_frames/frame_{timestamp}.jpg"
        
        # Create directory if it doesn't exist
        os.makedirs("hardware/captured_frames", exist_ok=True)
        
        cv2.imwrite(filename, frame)
        print(f"Frame saved: {filename}")
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        
        if self.camera:
            self.camera.close()
        elif hasattr(self, 'webcam'):
            self.webcam.release()
        
        cv2.destroyAllWindows()
        print("Real-time detection stopped.")

def main():
    """Main function to run real-time detection"""
    
    # Try to find the latest trained model
    import glob
    model_files = glob.glob("runs/detect/*/weights/best.pt")
    
    if model_files:
        latest_model = max(model_files, key=os.path.getctime)
        print(f"Found trained model: {latest_model}")
        detector = RealTimeSolarDefectDetector(model_path=latest_model)
    else:
        print("No trained model found. Using base model.")
        detector = RealTimeSolarDefectDetector()
    
    try:
        detector.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        detector.cleanup()

if __name__ == "__main__":
    main()
