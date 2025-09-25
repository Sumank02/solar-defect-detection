#!/usr/bin/env python3
"""
Test script for MLX90640 hardware integration pipeline
This script tests all components without requiring actual hardware
"""

import cv2
import numpy as np
import time
from mlx90640_camera import MLX90640Camera

def test_camera_pipeline():
    """Test the complete camera pipeline"""
    print("=== Testing MLX90640 Camera Pipeline ===")
    
    # Test 1: Camera initialization
    print("\n1. Testing camera initialization...")
    try:
        camera = MLX90640Camera()
        print("✅ Camera initialized successfully")
    except Exception as e:
        print(f"❌ Camera initialization failed: {e}")
        return False
    
    # Test 2: Frame capture
    print("\n2. Testing frame capture...")
    try:
        frame = camera.capture_frame()
        if frame is not None and frame.shape == (24, 32):
            print(f"✅ Frame captured successfully: {frame.shape}")
            print(f"   Temperature range: {np.min(frame):.1f}°C to {np.max(frame):.1f}°C")
        else:
            print(f"❌ Frame capture failed: {frame}")
            return False
    except Exception as e:
        print(f"❌ Frame capture failed: {e}")
        return False
    
    # Test 3: Thermal to visual conversion
    print("\n3. Testing thermal to visual conversion...")
    try:
        visual_frame = camera.thermal_to_visual(frame)
        if visual_frame is not None and visual_frame.shape == (24, 32, 3):
            print(f"✅ Visual conversion successful: {visual_frame.shape}")
        else:
            print(f"❌ Visual conversion failed: {visual_frame}")
            return False
    except Exception as e:
        print(f"❌ Visual conversion failed: {e}")
        return False
    
    # Test 4: YOLO resizing
    print("\n4. Testing YOLO resizing...")
    try:
        yolo_frame = camera.resize_for_yolo(visual_frame, target_size=640)
        if yolo_frame is not None and yolo_frame.shape == (640, 640, 3):
            print(f"✅ YOLO resizing successful: {yolo_frame.shape}")
        else:
            print(f"❌ YOLO resizing failed: {yolo_frame}")
            return False
    except Exception as e:
        print(f"❌ YOLO resizing failed: {e}")
        return False
    
    # Test 5: Temperature statistics
    print("\n5. Testing temperature statistics...")
    try:
        temp_stats = camera.get_temperature_stats(frame)
        if temp_stats and all(key in temp_stats for key in ['min_temp', 'max_temp', 'avg_temp', 'center_temp']):
            print("✅ Temperature statistics successful:")
            for key, value in temp_stats.items():
                print(f"   {key}: {value:.1f}°C")
        else:
            print(f"❌ Temperature statistics failed: {temp_stats}")
            return False
    except Exception as e:
        print(f"❌ Temperature statistics failed: {e}")
        return False
    
    # Test 6: Save test image
    print("\n6. Testing image saving...")
    try:
        # Create output directory
        import os
        os.makedirs("test_output", exist_ok=True)
        
        # Save thermal data as image
        cv2.imwrite("test_output/thermal_test.png", visual_frame)
        cv2.imwrite("test_output/yolo_test.png", yolo_frame)
        print("✅ Test images saved successfully")
    except Exception as e:
        print(f"❌ Image saving failed: {e}")
        return False
    
    # Test 7: Camera cleanup
    print("\n7. Testing camera cleanup...")
    try:
        camera.close()
        print("✅ Camera cleanup successful")
    except Exception as e:
        print(f"❌ Camera cleanup failed: {e}")
        return False
    
    print("\n=== All Tests Passed! ===")
    return True

def test_yolo_integration():
    """Test YOLO model integration"""
    print("\n=== Testing YOLO Integration ===")
    
    try:
        from ultralytics import YOLO
        import torch
        
        # Fix for PyTorch 2.8.0+ weights_only issue
        original_load = torch.load
        def safe_load(file, **kwargs):
            return original_load(file, weights_only=False, **kwargs)
        torch.load = safe_load
        
        # Try to load a model
        import glob
        model_files = glob.glob("../runs/detect/*/weights/best.pt")
        
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            print(f"Found trained model: {latest_model}")
            model = YOLO(latest_model)
            print("✅ YOLO model loaded successfully")
        else:
            print("No trained model found, using base model")
            model = YOLO('yolov8n.yaml')
            print("✅ Base YOLO model loaded successfully")
        
        # Test inference on a test frame
        camera = MLX90640Camera()
        frame = camera.capture_frame()
        visual_frame = camera.thermal_to_visual(frame)
        yolo_frame = camera.resize_for_yolo(visual_frame)
        
        # Run inference
        results = model(yolo_frame, conf=0.25)
        print(f"✅ YOLO inference successful: {len(results)} results")
        
        camera.close()
        return True
        
    except Exception as e:
        print(f"❌ YOLO integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting hardware integration tests...")
    
    # Test camera pipeline
    camera_ok = test_camera_pipeline()
    
    # Test YOLO integration
    yolo_ok = test_yolo_integration()
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY:")
    print(f"Camera Pipeline: {'✅ PASS' if camera_ok else '❌ FAIL'}")
    print(f"YOLO Integration: {'✅ PASS' if yolo_ok else '❌ FAIL'}")
    
    if camera_ok and yolo_ok:
        print("\n🎉 All tests passed! Hardware integration is ready.")
        print("You can now run:")
        print("  python simple_detection.py")
        print("  python real_time_detection.py")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
    
    print("="*50)
