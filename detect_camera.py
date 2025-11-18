import cv2
import sys
from ultralytics import YOLO
from collections import defaultdict
import time

# --- CẤU HÌNH ---
# 1. Đường dẫn đến "bộ não" AI đã huấn luyện
MODEL_PATH = 'runs/detect/train/weights/best.pt'

# 2. Đường dẫn đến camera
# === Tùy chọn A: Dùng camera IP ===
# Bạn cần tìm đường dẫn RTSP trong phần cài đặt camera an ninh của bạn
# RTSP_URL = 'rtsp://admin:password@192.168.1.100:554/stream1'

# === Tùy chọn B: Dùng Webcam (để thử nghiệm) ===
# RTSP_URL = 0  # Bỏ comment dòng này và comment dòng trên để dùng webcam
RTSP_URL = 'rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov' 


# 3. Ngưỡng tin cậy
CONF_THRESHOLD = 0.4
# --------------------

def run_camera_detection(model_path, video_source, conf_threshold):
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Lỗi tải mô hình: {e}")
        sys.exit()

    class_names = model.model.names
    
    # Kết nối đến nguồn video
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Lỗi: Không thể kết nối đến camera tại '{video_source}'")
        sys.exit()

    print("Đã kết nối camera! Bắt đầu nhận diện (Nhấn 'Esc' để thoát)...")

    while True:
        # Đọc từng khung hình
        success, frame = cap.read()
        if not success:
            print("Lỗi: Không thể đọc khung hình. Đang thử kết nối lại...")
            # Cố gắng kết nối lại
            cap.release()
            cap = cv2.VideoCapture(video_source)
            time.sleep(5)
            continue
        
        # Chạy AI trên khung hình
        results = model(frame, conf=conf_threshold, verbose=False) # verbose=False để tắt log
        result = results[0]
        counts = defaultdict(int)

        for box in result.boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
            class_id = int(box.cls[0])
            class_name = class_names.get(class_id, 'Unknown')
            confidence = float(box.conf[0])

            counts[class_name] += 1
            
            color = (0, 255, 0) if class_name == 'nhanvien' else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # In kết quả đếm ra Terminal (bạn có thể xóa nếu không muốn làm chậm)
        print(f"Phat hien: {dict(counts)}")

        # Hiển thị video
        cv2.imshow('AI Camera (Nhan Esc de thoat)', frame)
        
        # Nhấn 'Esc' để thoát
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
    # Dọn dẹp
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera_detection(MODEL_PATH, RTSP_URL, CONF_THRESHOLD)