from ultralytics import YOLO

# Img Dataset
Img_path = '/Users/seunghunjang/Desktop/Clothing_Classification/Dataset'
#Img_BGR_path = '/Users/seunghunjang/Desktop/Clothing_Classification/Dataset_BGR'

# Train Dir save Path
path = '/Users/seunghunjang/Desktop/Clothing_Classification/Train'

# Load Model
model = YOLO('yolov8m-cls.pt')


# Train the model
result = model.train(data=Img_path, epochs=25, imgsz=640)