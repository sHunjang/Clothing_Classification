from ultralytics import YOLO

# Load model
model = YOLO() # Pre trained model ex)best.pt

# Validate model
metrics = model.val()