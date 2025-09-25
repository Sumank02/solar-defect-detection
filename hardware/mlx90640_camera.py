import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import time
import os

# Try to import MLX90640 libraries - handle different possible implementations
try:
    # Try the most common MLX90640 library
    import board
    import busio
    import adafruit_mlx90640
    MLX90640_AVAILABLE = True
    print("Using Adafruit MLX90640 library")
except ImportError:
    try:
        # Try alternative MLX90640 library
        import smbus2 as smbus
        MLX90640_AVAILABLE = True
        print("Using SMBus MLX90640 library")
    except ImportError:
        MLX90640_AVAILABLE = False
        print("MLX90640 libraries not available - using simulation mode")

class MLX90640Camera:
    """Interface for MLX90640 IR camera (32x24 thermal sensor)"""
    
    def __init__(self, i2c_bus=1, use_simulation=False):
        """Initialize MLX90640 camera"""
        self.frame_shape = (24, 32)  # MLX90640 resolution
        self.use_simulation = use_simulation or not MLX90640_AVAILABLE
        
        if self.use_simulation:
            print("Running in simulation mode - generating synthetic thermal data")
            self.camera = None
        else:
            try:
                if 'adafruit_mlx90640' in globals():
                    # Use Adafruit library
                    i2c = busio.I2C(board.SCL, board.SDA)
                    self.camera = adafruit_mlx90640.MLX90640(i2c)
                    self.camera.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ
                    print("MLX90640 initialized with Adafruit library")
                else:
                    # Use SMBus library
                    self.bus = smbus.SMBus(i2c_bus)
                    self.camera = None
                    print("MLX90640 initialized with SMBus library")
            except Exception as e:
                print(f"Failed to initialize MLX90640: {e}")
                print("Falling back to simulation mode")
                self.use_simulation = True
                self.camera = None
    
    def capture_frame(self):
        """Capture a single thermal frame"""
        if self.use_simulation:
            return self._generate_simulation_frame()
        
        try:
            if 'adafruit_mlx90640' in globals() and self.camera:
                # Adafruit library
                frame = np.zeros((24, 32))
                self.camera.getFrame(frame)
                return frame
            elif hasattr(self, 'bus'):
                # SMBus library - basic implementation
                frame = np.zeros((24, 32))
                # Read from I2C address 0x33 (MLX90640 default)
                for i in range(24):
                    for j in range(32):
                        try:
                            # Read temperature data (simplified)
                            data = self.bus.read_word_data(0x33, 0x0400 + i * 32 + j)
                            temp = data * 0.02 - 273.15  # Convert to Celsius
                            frame[i, j] = temp
                        except:
                            frame[i, j] = 25.0  # Default temperature
                return frame
            else:
                return None
                
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return self._generate_simulation_frame()
    
    def _generate_simulation_frame(self):
        """Generate synthetic thermal data for testing"""
        # Create a realistic thermal pattern
        frame = np.zeros((24, 32))
        
        # Add some "hot spots" (simulating solar panel defects)
        for i in range(24):
            for j in range(32):
                # Base temperature around 25°C
                base_temp = 25.0
                
                # Add some variation
                variation = np.sin(i * 0.3) * np.cos(j * 0.3) * 5
                
                # Add hot spots (defects)
                if (i-12)**2 + (j-16)**2 < 16:  # Center hot spot
                    frame[i, j] = base_temp + variation + 15
                elif (i-6)**2 + (j-8)**2 < 9:   # Top-left hot spot
                    frame[i, j] = base_temp + variation + 20
                elif (i-18)**2 + (j-24)**2 < 9: # Bottom-right hot spot
                    frame[i, j] = base_temp + variation + 18
                else:
                    frame[i, j] = base_temp + variation
        
        return frame
    
    def thermal_to_visual(self, thermal_array, colormap='hot'):
        """Convert thermal data to visual image"""
        if thermal_array is None:
            return None
            
        # Normalize thermal data to 0-255
        temp_min = np.min(thermal_array)
        temp_max = np.max(thermal_array)
        
        if temp_max > temp_min:
            normalized = ((thermal_array - temp_min) / (temp_max - temp_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(thermal_array, dtype=np.uint8)
        
        # Apply colormap
        if colormap == 'hot':
            colored = cv2.applyColorMap(normalized, cv2.COLORMAP_HOT)
        elif colormap == 'jet':
            colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        else:
            colored = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
        
        return colored
    
    def resize_for_yolo(self, image, target_size=640):
        """Resize image to YOLO input size"""
        if image is None:
            return None
            
        # Resize to target size (maintain aspect ratio)
        height, width = image.shape[:2]
        
        # Calculate scaling factor
        scale = target_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height))
        
        # Create square canvas with padding
        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
        # Center the resized image
        y_offset = (target_size - new_height) // 2
        x_offset = (target_size - new_width) // 2
        
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return canvas
    
    def get_temperature_stats(self, thermal_array):
        """Get temperature statistics from thermal data"""
        if thermal_array is None:
            return None
            
        return {
            'min_temp': np.min(thermal_array),
            'max_temp': np.max(thermal_array),
            'avg_temp': np.mean(thermal_array),
            'center_temp': thermal_array[12, 16]  # Center pixel
        }
    
    def close(self):
        """Close camera connection"""
        try:
            if hasattr(self, 'bus'):
                self.bus.close()
            if self.camera:
                if hasattr(self.camera, 'close'):
                    self.camera.close()
        except:
            pass
