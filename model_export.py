from ultralytics import YOLO
import onnx

# Load model
model = YOLO('/opt/homebrew/runs/classify/train8/weights/best.pt')

# Export model
model.export(format='onnx')