import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json

# ================== LOAD MODEL ==================
MODEL_PATH = "models/best_model.h5"
CLASS_PATH = "models/class_names.json"

model = load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

# ================== CONFIG ==================
CONF_THRESHOLD = 0.75
HSV_THRESHOLD = 0.20

# ================== HSV GATE ==================
def hsv_gate_check(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, np.array([10, 100, 60]), np.array([90, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([120, 60, 50]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    ratio = np.sum(mask > 0) / (frame.shape[0] * frame.shape[1])
    return ratio > HSV_THRESHOLD, ratio


# ================== CAMERA ==================
cap = cv2.VideoCapture(0)

print("🎥 Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    is_fruit, ratio = hsv_gate_check(frame)

    img = cv2.resize(frame, (224, 224))
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)
    conf = np.max(preds)
    label = CLASS_NAMES[np.argmax(preds)]

    if not is_fruit and conf < 0.95:
        text = "Not a fruit"
        color = (0, 0, 255)
    elif conf < CONF_THRESHOLD:
        text = "Unknown object"
        color = (0, 255, 255)
    else:
        text = f"{label} ({conf:.2f})"
        color = (0, 255, 0)

    cv2.putText(display, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(display, f"HSV ratio: {ratio:.2f}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Fruit Recognition", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
