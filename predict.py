from ultralytics import YOLO
import torch
import glob
import os

def predict_with_model():
	"""Predict using the trained model with PyTorch compatibility fix"""
	
	# Fix for PyTorch 2.8.0+ weights_only issue
	original_load = torch.load
	def safe_load(file, **kwargs):
		return original_load(file, weights_only=False, **kwargs)
	torch.load = safe_load
	
	try:
		# Resolve latest trained weight automatically
		candidates = glob.glob(os.path.join('runs', 'detect', '*', 'weights', 'best.pt'))
		if candidates:
			best_path = max(candidates, key=os.path.getmtime)
			print(f"Using trained model: {best_path}")
			model = YOLO(best_path)
		else:
			# Known fallbacks from project history
			fallbacks = [
				os.path.join('runs', 'detect', 'solar_defect_train9', 'weights', 'best.pt'),
				os.path.join('runs', 'detect', 'solar_defect_train', 'weights', 'best.pt'),
			]
			for p in fallbacks:
				if os.path.exists(p):
					print(f"Using fallback trained model: {p}")
					model = YOLO(p)
					break
			else:
				# Final fallback: local small weights or yaml
				local_pt = os.path.join(os.getcwd(), 'yolov8n.pt')
				if os.path.exists(local_pt):
					print(f"No trained weights found. Falling back to local weights: {local_pt}")
					model = YOLO(local_pt)
				else:
					print("No trained weights found. Falling back to base config 'yolov8n.yaml'")
					model = YOLO('yolov8n.yaml')
		
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