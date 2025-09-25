import os
import glob
import shutil
from typing import List, Optional, Tuple, Set

import cv2
import torch
from ultralytics import YOLO

# Resolve directories relative to this file inside catalog/scripts
FILE_DIR = os.path.dirname(__file__)
CATALOG_DIR = os.path.normpath(os.path.join(FILE_DIR, ".."))
PROJECT_ROOT = os.path.normpath(os.path.join(CATALOG_DIR, ".."))

# Project-level dataset and runs
DATA_VALID = os.path.join(PROJECT_ROOT, "dataset", "valid")
DATA_TRAIN = os.path.join(PROJECT_ROOT, "dataset", "train")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

# Catalog outputs
OUTPUT_DIR = os.path.join(CATALOG_DIR, "docs", "images")
RUNS_NAME = "catalog_examples"

# Classes (must match dataset/data.yaml)
CLASS_NAMES: List[str] = [
	"MultiByPassed",
	"MultiDiode",
	"MultiHotSpot",
	"SingleByPassed",
	"SingleDiode",
	"SingleHotSpot",
	"StringOpenCircuit",
	"StringReversedPolarity",
]

DATASETS = [
	(os.path.normpath(DATA_VALID), "valid"),
	(os.path.normpath(DATA_TRAIN), "train"),
]


def find_latest_best() -> Optional[str]:
	candidates = glob.glob(os.path.join(RUNS_DIR, "detect", "*", "weights", "best.pt"))
	return max(candidates, key=os.path.getmtime) if candidates else None


def resolve_image_path(images_dir: str, label_path: str) -> Optional[str]:
	base = os.path.basename(label_path).rsplit(".", 1)[0]
	for ext in (".jpg", ".jpeg", ".png"):
		candidate = os.path.join(images_dir, base + ext)
		if os.path.exists(candidate):
			return candidate
	return None


def parse_label_file(label_path: str) -> List[Tuple[int, float, float, float, float]]:
	boxes: List[Tuple[int, float, float, float, float]] = []
	with open(label_path, "r", encoding="utf-8") as f:
		for line in f:
			parts = line.strip().split()
			if len(parts) >= 5 and parts[0].isdigit():
				cls = int(parts[0])
				xc, yc, w, h = map(float, parts[1:5])
				boxes.append((cls, xc, yc, w, h))
	return boxes


def draw_gt_boxes(img_path: str, label_path: str, focus_cls: int):
	img = cv2.imread(img_path)
	if img is None:
		return None
	H, W = img.shape[:2]
	boxes = parse_label_file(label_path)
	for cls, xc, yc, w, h in boxes:
		x1 = int((xc - w / 2) * W)
		y1 = int((yc - h / 2) * H)
		x2 = int((xc + w / 2) * W)
		y2 = int((yc + h / 2) * H)
		if cls == focus_cls:
			color, th = (0, 255, 0), 2
			label = f"{CLASS_NAMES[cls]}"
		else:
			color, th = (128, 128, 128), 1
			label = f"{CLASS_NAMES[cls]}"
		cv2.rectangle(img, (x1, y1), (x2, y2), color, th)
		cv2.putText(img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
	return img


def collect_image_for_class(class_index: int, used_images: Set[str]) -> Optional[Tuple[str, str]]:
	for root, _ in DATASETS:
		labels_dir = os.path.join(root, "labels")
		images_dir = os.path.join(root, "images")
		for txt_path in glob.glob(os.path.join(labels_dir, "*.txt")):
			try:
				boxes = parse_label_file(txt_path)
				if any(b[0] == class_index for b in boxes):
					img_path = resolve_image_path(images_dir, txt_path)
					if img_path and img_path not in used_images:
						return img_path, txt_path
			except Exception:
				continue
	for root, _ in DATASETS:
		labels_dir = os.path.join(root, "labels")
		images_dir = os.path.join(root, "images")
		for txt_path in glob.glob(os.path.join(labels_dir, "*.txt")):
			try:
				boxes = parse_label_file(txt_path)
				if any(b[0] == class_index for b in boxes):
					img_path = resolve_image_path(images_dir, txt_path)
					if img_path:
						return img_path, txt_path
			except Exception:
				continue
	return None


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def main() -> None:
	print("Preparing defect catalog examples...")

	_original_load = torch.load
	def _safe_load(file, **kwargs):
		return _original_load(file, weights_only=False, **kwargs)
	torch.load = _safe_load

	best = find_latest_best()
	model = YOLO(best) if best and os.path.exists(best) else YOLO("yolov8n.yaml")
	if best:
		print(f"Using trained model: {best}")
	else:
		print("No trained model found. Using base model (only GT boxes will be visible if model misses).")

	ensure_dir(OUTPUT_DIR)

	used: Set[str] = set()
	missing: List[str] = []
	for idx, cls in enumerate(CLASS_NAMES):
		print(f"Selecting example for class {idx}: {cls}...")
		pair = collect_image_for_class(idx, used)
		if not pair:
			print(f"- No image found for class {cls} in valid/train.")
			missing.append(cls)
			continue
		img_path, label_path = pair
		used.add(img_path)
		print(f"- Using image: {img_path}")

		annotated = draw_gt_boxes(img_path, label_path, idx)
		if annotated is None:
			print(f"- Failed to load/draw: {img_path}")
			missing.append(cls)
			continue

		try:
			res = model.predict(source=img_path, save=False, conf=0.20, imgsz=640, verbose=False)
			if res and len(res) > 0 and getattr(res[0], "boxes", None) is not None:
				for b in res[0].boxes:
					x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
					cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
					cv2.putText(annotated, f"pred", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1, cv2.LINE_AA)
		except Exception:
			pass

		dst = os.path.join(OUTPUT_DIR, f"{cls}.jpg")
		cv2.imwrite(dst, annotated)
		print(f"- Saved example: {dst}")

	if missing:
		print("\nThe following classes could not be illustrated (no examples found):")
		for cls in missing:
			print(f"- {cls}")
	else:
		print("\nAll classes illustrated successfully.")

	print(f"\nDone. Check folder: {OUTPUT_DIR}")

if __name__ == "__main__":
	main()
