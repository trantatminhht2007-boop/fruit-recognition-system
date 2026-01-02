import tensorflow as tf
from data_loader import load_datasets
from model import build_model
import os

if not os.path.exists('models'): os.makedirs('models')

# 1. Load Data
train_ds, val_ds, class_names = load_datasets("dataset")

# 2. Data Augmentation (Giữ nguyên cấu hình mạnh mẽ của bạn)
augmentation_layer = tf.keras.Sequential([
    tf.keras.layers.RandomCrop(height=224, width=224), 
    tf.keras.layers.Resizing(224, 224), 
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.3), 
    tf.keras.layers.RandomZoom(0.3),     
    tf.keras.layers.RandomTranslation(0.15, 0.15), 
    tf.keras.layers.RandomContrast(0.4), 
    tf.keras.layers.RandomBrightness(0.4),
    tf.keras.layers.GaussianNoise(stddev=0.1)
])
train_ds = train_ds.map(lambda x, y: (augmentation_layer(x, training=True), y))

# 3. Khởi tạo Model
model = build_model()

# --- GIAI ĐOẠN 1: Transfer Learning (Huấn luyện lớp phân loại mới) ---
print(">>> Giai đoạn 1: Huấn luyện lớp phân loại...")
callbacks1 = [
    tf.keras.callbacks.ModelCheckpoint('models/best_model_stage1.h5', monitor='val_accuracy', save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
]
model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks1)

# --- GIAI ĐOẠN 2: Fine-Tuning (Mở khóa não bộ MobileNetV2) ---
print(">>> Giai đoạn 2: Fine-Tuning toàn mạng...")

# Tìm lớp base_model (thường là lớp thứ 1 hoặc có tên cụ thể)
# Ở đây tôi giả định lớp đầu tiên là base_model của bạn
base_model = model.layers[1] 
base_model.trainable = True

# Chỉ mở khóa 30 lớp cuối để tránh phá hủy kiến thức cũ
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Re-compile với Learning Rate cực thấp
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks2 = [
    tf.keras.callbacks.ModelCheckpoint('models/best_model.h5', monitor='val_accuracy', save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks2)