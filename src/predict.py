import cv2
import numpy as np
import tensorflow as tf
import json

# Ngưỡng cấu hình
HSV_AREA_THRESHOLD = 0.15
AI_CONF_THRESHOLD = 0.75
SAFE_FALLBACK_THRESHOLD = 0.95

# Load class names từ JSON (đồng bộ với train.py)
CLASS_NAMES = ['apple', 'banana', 'grapes', 'guava', 'orange']
try:
    with open('models/class_names.json', 'r', encoding='utf-8') as f:
        CLASS_NAMES = json.load(f)
    print(f"✅ Loaded class names: {CLASS_NAMES}")
except FileNotFoundError:
    print(f"⚠️ class_names.json not found, using default: {CLASS_NAMES}")

def hsv_gate_check(frame):
    """
    Kiểm tra xem frame có chứa trái cây không dựa trên HSV color space.
    
    Returns:
        is_fruit (bool): True nếu phát hiện màu sắc trái cây
        ratio (float): Tỷ lệ pixel màu trái cây
    """
    if frame is None or frame.size == 0:
        return False, 0.0
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Nhóm 1: Sáng (Cam, Chuối, Táo xanh)
    mask1 = cv2.inRange(hsv, np.array([10, 100, 60]), np.array([90, 255, 255]))
    
    # Nhóm 2: Tối (Nho đen, Táo đỏ)
    mask2 = cv2.inRange(hsv, np.array([120, 100, 30]), np.array([180, 255, 255]))
    
    # FIXED: Thêm dải đỏ cho táo đỏ, dâu (Hue wrap around 0-180)
    mask_red1 = cv2.inRange(hsv, np.array([0, 100, 60]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([170, 100, 60]), np.array([180, 255, 255]))
    
    # Combine tất cả masks
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.bitwise_or(mask, cv2.bitwise_or(mask_red1, mask_red2))
    
    # Khử nhiễu nhẹ (optional - giúp ổn định hơn)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    ratio = np.count_nonzero(mask) / (frame.shape[0] * frame.shape[1])
    return ratio > HSV_AREA_THRESHOLD, ratio

def preprocess_image(frame):
    """
    Resize và chuẩn bị ảnh cho model.
    Không cần normalize vì model đã nhúng preprocessing.
    """
    img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    img = np.expand_dims(img.astype(np.float32), axis=0)
    return img

def predict_fruit(model, frame):
    """
    Predict trái cây với Double Gate: HSV + Softmax confidence.
    
    Args:
        model: Trained Keras model
        frame: BGR image from OpenCV
    
    Returns:
        label (str): Tên trái cây hoặc "Not a fruit"/"Unknown Object"
        score (float): Confidence score [0-1]
    """
    is_fruit, ratio = hsv_gate_check(frame)
    img = preprocess_image(frame)
    
    # OPTIMIZED: Dùng model() thay vì predict() → nhanh hơn 3-5x
    preds = model(img, training=False).numpy()
    score = float(np.max(preds))
    label = CLASS_NAMES[int(np.argmax(preds))]
    
    # Double Gate Logic
    if score > SAFE_FALLBACK_THRESHOLD:
        # Confidence cực cao → tin AI 100%
        return label, score
    
    if not is_fruit:
        # HSV không detect màu trái cây → reject
        return "Not a fruit", score
    
    if score < AI_CONF_THRESHOLD:
        # Confidence thấp → unknown
        return "Unknown Object", score
    
    return label, score

# ===== DEMO: Real-time webcam inference =====
if __name__ == "__main__":
    try:
        # Load model
        print("Loading model...")
        model = tf.keras.models.load_model('models/best_model.h5')
        print("✅ Model loaded successfully")
        
        # OPTIONAL: Compile với XLA để nhanh hơn
        # model.compile(jit_compile=True)
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        print("📷 Webcam opened. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            # Predict
            label, score = predict_fruit(model, frame)
            
            # Display result
            color = (0, 255, 0) if label not in ["Not a fruit", "Unknown Object"] else (0, 0, 255)
            cv2.putText(frame, f"{label}: {score:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.imshow("Fruit Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
    except FileNotFoundError:
        print("❌ Model file not found. Please train the model first by running train.py")
    except Exception as e:
        print(f"❌ Error: {e}")