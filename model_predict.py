from ultralytics import YOLO

# Load pretrained model
model = YOLO('/opt/homebrew/runs/classify/train2/weights/best.pt')

predic_BGR_img_path = '/Users/seunghunjang/Desktop/Clothing_Classification/Predict_dataset/BGR'
predic_img_path = '/Users/seunghunjang/Desktop/Clothing_Classification/Predict_dataset/BGR_X'

# Predict image
result = model.predict(source=predic_BGR_img_path, save=True)