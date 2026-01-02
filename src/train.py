import tensorflow as tf
from data_loader import load_datasets
from model import build_model
import os

# Tạo thư mục lưu trữ
if not os.path.exists('models'): os.makedirs('models')

# 1. Load Data
train_ds, val_ds, class_names = load_datasets("dataset")
# 2. Data Augmentation Pipeline (Tăng cường dữ liệu)
augmentation_layer = tf.keras.Sequential([
    # 1. Loại bỏ nhiễu rìa (Nút bấm, UI artifacts)
    # Cắt CenterCrop sâu hơn (85%) để chắc chắn mất các icon ở góc
    tf.keras.layers.CenterCrop(height=200, width=200), 
    tf.keras.layers.Resizing(224, 224), 
    
    # 2. Đa dạng hóa góc nhìn (Xoay, Lật, Dịch chuyển) 
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.3), # Tăng lên 0.3 để chấp nhận mọi góc cầm tay 
    tf.keras.layers.RandomZoom(0.3),     # Giả lập việc đưa quả lại gần/ra xa camera 
    tf.keras.layers.RandomTranslation(0.15, 0.15), # Quan trọng: Giúp nhận diện khi quả không nằm giữa 
    
    # 3. Xử lý ánh sáng (Khắc phục lỗi camera tối/chói) 
    # Tăng Contrast và Brightness lên 0.4 để mô hình quen với ảnh webcam thực tế 
    tf.keras.layers.RandomContrast(0.4), 
    tf.keras.layers.RandomBrightness(0.4)
])
train_ds = train_ds.map(lambda x, y: (augmentation_layer(x, training=True), y))

# 3. Khởi tạo Model
model = build_model()

# 4. Thiết lập bộ 3 Callbacks "Quyền lực"
callbacks = [
    # Lưu bản tốt nhất dựa trên độ chính xác
    tf.keras.callbacks.ModelCheckpoint('models/best_model.h5', monitor='val_accuracy', save_best_only=True),
    # Dừng khi không còn tiến bộ để tránh học vẹt
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    # Giảm LR khi học chậm lại để tối ưu hóa
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
]

# 5. Huấn luyện
model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=callbacks)