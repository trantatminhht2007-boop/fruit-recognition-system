import cv2
import numpy as np
import tensorflow as tf

class_names = ['apple', 'banana', 'grapes', 'guava', 'orange']

def hsv_gate_check(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Nhóm 1: Sáng (Cam, Chuối, Táo xanh)
    mask1 = cv2.inRange(hsv, np.array([10, 100, 60]), np.array([90, 255, 255]))
    # Nhóm 2: Tối (Nho đen, Táo đỏ) - S tăng lên 100 để lọc nền đen
    mask2 = cv2.inRange(hsv, np.array([120, 100, 30]), np.array([180, 255, 255]))
    
    mask = cv2.bitwise_or(mask1, mask2)
    ratio = np.sum(mask > 0) / (frame.shape[0] * frame.shape[1])
    return ratio > 0.15, ratio # Hạ xuống 0.15 cho linh hoạt

def predict_fruit(model, frame):
    is_fruit, ratio = hsv_gate_check(frame)
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)
    
    preds = model.predict(img, verbose=0)
    score = np.max(preds)
    label = class_names[np.argmax(preds)]
    
    # Cổng gác kép
    if not is_fruit and score < 0.95: return "Not a fruit", score
    if score < 0.75: return "Unknown Object", score
    return label, score