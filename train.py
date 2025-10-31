from ultralytics import YOLO

# Tải một mô hình YOLOv8 gốc (pre-trained)
model = YOLO('yolov8n.pt')

# Bắt đầu huấn luyện
# 'data.yaml' là file bạn tải từ Roboflow
results = model.train(data='data.yaml', 
                      epochs=100, 
                      imgsz=640)

# Mô hình tốt nhất sẽ được lưu tại: runs/detect/train/weights/best.pt
print("Hoàn tất huấn luyện!")