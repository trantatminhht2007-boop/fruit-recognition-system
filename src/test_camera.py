import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ================= CONFIG =================
MODEL_PATH = "models/best_model.keras"
CLASS_NAMES = ['apple', 'banana', 'grapes', 'guava', 'orange']

HSV_THRESHOLD = 0.20
CONF_THRESHOLD = 0.75
# =========================================


# ---------- Load model ----------
print("Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")


# ---------- HSV GATE ----------
def hsv_gate(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, (10, 100, 60), (90, 255, 255))
    mask2 = cv2.inRange(hsv, (120, 60, 50), (180, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    ratio = np.sum(mask > 0) / (frame.shape[0] * frame.shape[1])
    return ratio > HSV_THRESHOLD, ratio


# ---------- CAMERA ----------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("🎥 Camera started — Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------- Preprocess ----------
    img = cv2.resize(frame, (224, 224))
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)
    score = np.max(preds)
    label = CLASS_NAMES[np.argmax(preds)]

    is_fruit, ratio = hsv_gate(frame)

    if not is_fruit and score < 0.95:
        text = "Not a fruit"
        color = (0, 0, 255)
    elif score < CONF_THRESHOLD:
        text = "Unknown"
        color = (0, 255, 255)
    else:
        text = f"{label} ({score:.2f})"
        color = (0, 255, 0)

    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Fruit Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
