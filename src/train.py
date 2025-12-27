import tensorflow as tf
from data_loader import load_datasets
from model import build_model
import os

# Tạo thư mục lưu trữ
if not os.path.exists('models'): os.makedirs('models')

# 1. Load Data
train_ds, val_ds, class_names = load_datasets("dataset")

# 2. Data Augmentation Pipeline (Thực hiện ngoài Model - tf.data)
augmentation_layer = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
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