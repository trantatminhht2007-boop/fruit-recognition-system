import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. Đồng bộ nhãn
# Leader hãy kiểm tra thứ tự này bằng lệnh: print(train_ds.class_names)
class_names = ['apple', 'banana', 'grapes', 'guava', 'orange']

def hsv_gate_check(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Gợi ý từ ChatGPT: Tách dải màu để lọc nhiễu tốt hơn
    # Nhóm 1: Cam, Vàng, Xanh lục (Cam, Chuối, Ổi, Táo xanh)
    lower1 = np.array([10, 100, 60])
    upper1 = np.array([90, 255, 255])
    
    # Nhóm 2: Tím, Đỏ (Táo đỏ, Nho tím/đen)
    lower2 = np.array([120, 60, 50]) # V > 50 để bỏ qua nền đen 3D
    upper2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Khử nhiễu
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    ratio = np.sum(mask > 0) / (frame.shape[0] * frame.shape[1])
    
    # Ngưỡng 0.2: Cân bằng giữa webcam và ảnh zoom
    return ratio > 0.2, ratio

def predict_fruit(model, frame):
    is_fruit, ratio = hsv_gate_check(frame)
    
    # Xử lý ảnh cho MobileNetV2 (Sửa lỗi dải 0-1)
    img = cv2.resize(frame, (224, 224))
    img = preprocess_input(img.astype(np.float32)) # Chuẩn hóa [-1, 1]
    img = np.expand_dims(img, axis=0)
    
    preds = model.predict(img)
    score = np.max(preds)
    label = class_names[np.argmax(preds)]
    
    # CÔNG GÁC KÉP: Kết hợp HSV Gate và Softmax Threshold
    # Thêm "Soft fallback" nếu model cực kỳ tự tin (>95%)
    if not is_fruit and score < 0.95:
        return "Not a fruit", score
    
    if score < 0.75:
        return "Unknown Object", score
        
    return label, score