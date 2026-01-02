import tensorflow as tf
from data_loader import load_datasets
from model import build_model
import os

# 1. Load Data
train_ds, val_ds, class_names = load_datasets("dataset")

# 2. Augmentation mạnh mẽ cho ảnh Hybrid
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomCrop(height=224, width=224),
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomContrast(0.4),
    tf.keras.layers.GaussianNoise(stddev=0.1) # Thêm nhiễu camera
])
train_ds = train_ds.map(lambda x, y: (augmentation(x, training=True), y))

model = build_model()

# Giai đoạn 1: Học đặc trưng cơ bản
print(">>> Giai đoạn 1: Training Top Layers...")
model.fit(train_ds, validation_data=val_ds, epochs=15)

# Giai đoạn 2: Fine-tuning mở khóa 30 lớp cuối
print(">>> Giai đoạn 2: Fine-Tuning...")
base_model = model.get_layer("mobilenetv2_1.00_224")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # LR cực thấp
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_ds, validation_data=val_ds, epochs=15)
model.save('models/best_model.h5')