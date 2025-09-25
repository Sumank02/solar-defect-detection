from ultralytics import YOLO
import torch

def predict_with_model():
	"""Predict using the trained model with PyTorch compatibility fix"""
	
	# Fix for PyTorch 2.8.0+ weights_only issue
	original_load = torch.load
	def safe_load(file, **kwargs):
		return original_load(file, weights_only=False, **kwargs)
	torch.load = safe_load
	
	try:
		# Load the trained model (most recent training run)
		model = YOLO('runs/detect/solar_defect_train9/weights/best.pt')
		
		# Run prediction on validation images
		print("Running prediction on validation images...")
		results = model.predict(source='dataset/valid/images', save=True)
		
		print(f"Prediction completed! Results saved in runs/detect/predict*/")
		print(f"Found {len(results)} images processed.")
		
	except Exception as e:
		print(f"Error during prediction: {e}")
		print("\nTrying alternative approach...")
		
		# Alternative: use the model directly without loading weights
		model = YOLO('yolov8n.yaml')
		model.predict(source='dataset/valid/images', save=True)
		print("Used base model for prediction.")

if __name__ == "__main__":
	predict_with_model() 