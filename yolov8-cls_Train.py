from ultralytics import YOLO

# Img Dataset
Img_path = '/Users/seunghunjang/Desktop/Clothing_Classification/Dataset'
#Img_BGR_path = '/Users/seunghunjang/Desktop/Clothing_Classification/Dataset_BGR'


# Load Model
model = YOLO('yolov8s-cls.pt')

# pretrained model
# model = YOLO('best.pt')


# Train the model
result = model.train(data=Img_path, epochs=1, imgsz=480, cache=True)