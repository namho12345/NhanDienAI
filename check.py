import cv2
import sys
from ultralytics import YOLO
from collections import defaultdict

MODEL_PATH = 'runs/detect/train/weights/best.pt'
IMAGE_PATH = 'D:/AI/NhanDienAI/train/images/PXL-3_193_jpg.rf.7ef4af8d29bf4608a186a83223f763ea.jpg'
CONF_THRESHOLD = 0.4        
def detect_objects(model_path, image_path, conf_threshold):
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Lỗi: Không thể tải mô hình từ '{model_path}'.")
        print(f"File best.pt có tồn tại không? Lỗi chi tiết: {e}")
        sys.exit()
    class_names = model.model.names 
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Không tìm thấy hoặc không thể đọc ảnh: {image_path}")
    except Exception as e:
        print(e)
        sys.exit()
    results = model(img, conf=conf_threshold)
    result = results[0] 
    counts = defaultdict(int)

    for box in result.boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
        class_id = int(box.cls[0])
        class_name = class_names.get(class_id, 'Unknown')
        confidence = float(box.conf[0])
        counts[class_name] += 1
        color = (0, 255, 0) if class_name == 'nhanvien' else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name}: {confidence:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    print("--- BÁO CÁO NHẬN DIỆN ---")
    if not counts:
        print("Không phát hiện được đối tượng nào.")
    else:
        for class_name, count in counts.items():
            print(f"Tổng số '{class_name}': {count}")
    print("--------------------------")
    
    cv2.imshow('Ket qua Nhan dien', img)
    
    cv2.imwrite('result_image.jpg', img)
    print(f"Đã lưu ảnh kết quả vào 'result_image.jpg'")
    while True:
        if cv2.waitKey(1) & 0xFF == 27: 
            break
            
    cv2.destroyAllWindows()
if __name__ == "__main__":
    detect_objects(MODEL_PATH, IMAGE_PATH, CONF_THRESHOLD)