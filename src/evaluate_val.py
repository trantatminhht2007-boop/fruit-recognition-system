import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tabulate import tabulate # Cài đặt: pip install tabulate

# Cấu hình giống hệt file predict.py
HSV_THRESHOLD = 0.2
CONF_THRESHOLD = 0.75
CLASS_NAMES = ['apple', 'banana', 'grapes', 'guava', 'orange']
VAL_PATH = "dataset/val"
MODEL_PATH = "models/best_model.h5"

def evaluate():
    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)
    results = {name: {"correct": 0, "total": 0, "rejected_hsv": 0, "low_conf": 0} for name in CLASS_NAMES}

    print(f"--- Đang bắt đầu đánh giá tập VAL tại {VAL_PATH} ---")

    for label in CLASS_NAMES:
        folder_path = os.path.join(VAL_PATH, label)
        if not os.path.exists(folder_path): continue

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            frame = cv2.imread(img_path)
            if frame is None: continue

            results[label]["total"] += 1

            # 1. Kiểm tra HSV Gate
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, np.array([10, 100, 60]), np.array([90, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([120, 60, 50]), np.array([180, 255, 255]))
            mask = cv2.bitwise_or(mask1, mask2)
            ratio = np.sum(mask > 0) / (frame.shape[0] * frame.shape[1])

            if ratio < HSV_THRESHOLD:
                results[label]["rejected_hsv"] += 1
                continue

            # 2. Dự đoán với MobileNetV2
            img = cv2.resize(frame, (224, 224))
            img = preprocess_input(img.astype(np.float32))
            img = np.expand_dims(img, axis=0)
            
            preds = model.predict(img, verbose=0)
            score = np.max(preds)
            pred_label = CLASS_NAMES[np.argmax(preds)]

            # 3. Kiểm tra Softmax Gate
            if score < CONF_THRESHOLD:
                results[label]["low_conf"] += 1
            elif pred_label == label:
                results[label]["correct"] += 1

    # Xuất báo cáo dạng bảng
    table_data = []
    for name, stat in results.items():
        acc = (stat["correct"] / stat["total"] * 100) if stat["total"] > 0 else 0
        table_data.append([name, stat["total"], stat["correct"], stat["rejected_hsv"], stat["low_conf"], f"{acc:.1f}%"])

    headers = ["Loại quả", "Tổng ảnh", "Đúng", "Lỗi HSV", "Lỗi Conf", "Accuracy"]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    evaluate()