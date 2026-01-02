import tensorflow as tf
from data_loader import load_datasets
from model import build_model
import json
import os

# Tạo thư mục models nếu chưa có
os.makedirs("models", exist_ok=True)

# 1. Load Data với seed để reproducible
train_ds, val_ds, class_names = load_datasets("dataset", img_size=(224, 224), batch_size=32, seed=42)

# 2. Augmentation "lỳ" và hữu ích
augmentation = tf.keras.Sequential([
    # FIXED: Thay RandomCrop → RandomZoom (có ích hơn với ảnh 224x224)
    tf.keras.layers.RandomZoom(height_factor=(-0.2, 0.1)),  # Zoom in/out
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.3),  # ±30%
    tf.keras.layers.RandomContrast(0.4),
    tf.keras.layers.RandomBrightness(factor=0.4),  # THÊM: Tăng độ "lỳ" với ánh sáng
    tf.keras.layers.GaussianNoise(stddev=0.1)  # Mô phỏng nhiễu camera
], name="augmentation")

train_ds = train_ds.map(lambda x, y: (augmentation(x, training=True), y))

# 3. Build model
model = build_model(num_classes=len(class_names))

# 4. Callbacks để tối ưu training
callbacks = [
    # Dừng sớm nếu val_accuracy không cải thiện sau 5 epochs
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    # Lưu best model
    tf.keras.callbacks.ModelCheckpoint(
        'models/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    # Giảm learning rate khi val_loss plateau
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# ===== GIAI ĐOẠN 1: Training Top Layers =====
print("\n" + "="*60)
print(">>> GIAI ĐOẠN 1: Training Top Layers (15 epochs)")
print("="*60)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks,
    verbose=1
)

# ===== GIAI ĐOẠN 2: Fine-Tuning =====
print("\n" + "="*60)
print(">>> GIAI ĐOẠN 2: Fine-Tuning 30 lớp cuối (15 epochs)")
print("="*60)

# FIXED: Dùng tên đã đặt trong model.py
base_model = model.get_layer("mobilenetv2_1.00_224")
base_model.trainable = True

# Chỉ mở khóa 30 lớp cuối
for layer in base_model.layers[:-30]:
    layer.trainable = False

print(f"✅ Unfrozen last 30 layers of MobileNetV2")
print(f"✅ Trainable params: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

# Compile lại với learning rate cực thấp
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks,
    verbose=1
)

# 5. Lưu model và class names
model.save('models/best_model.h5')
print(f"✅ Model saved to models/best_model.h5")

# THÊM: Lưu class_names.json để predict.py đồng bộ
with open('models/class_names.json', 'w', encoding='utf-8') as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)
print(f"✅ Class names saved to models/class_names.json: {class_names}")

# In kết quả cuối
print("\n" + "="*60)
print("TRAINING COMPLETED!")
print("="*60)
final_train_acc = history2.history['accuracy'][-1]
final_val_acc = history2.history['val_accuracy'][-1]
print(f"Final Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")
print(f"Overfitting Gap: {abs(final_train_acc - final_val_acc):.4f}")
print("="*60)