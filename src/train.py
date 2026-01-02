import tensorflow as tf
from data_loader import load_datasets
from model import build_model
import json
import os

# =========================
# CONFIG
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS_STAGE1 = 15
EPOCHS_STAGE2 = 15
MODEL_PATH = "models/best_model.keras"

os.makedirs("models", exist_ok=True)

# =========================
# LOAD DATA
# =========================
train_ds, val_ds, class_names = load_datasets(
    "dataset",
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

# =========================
# DATA AUGMENTATION
# =========================
augmentation = tf.keras.Sequential([
    # FIXED: Thay RandomCrop → RandomZoom (có ích hơn với ảnh 224x224)
    tf.keras.layers.RandomZoom(height_factor=(-0.2, 0.1)),  # Zoom in/out
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.3),  # ±30%
    tf.keras.layers.RandomContrast(0.3),
    tf.keras.layers.RandomBrightness(factor=0.3),  # THÊM: Tăng độ "lỳ" với ánh sáng
    tf.keras.layers.GaussianNoise(stddev=0.05)  # THÊM: Nhiễu Gaussian nhẹ
], name="augmentation")

train_ds = train_ds.map(lambda x, y: (augmentation(x, training=True), y))

# =========================
# BUILD MODEL
# =========================
model = build_model(num_classes=len(class_names))

# =========================
# CALLBACKS
# =========================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# =========================
# PHASE 1 — TRAIN CLASSIFIER
# =========================
print("\n" + "=" * 60)
print(">>> PHASE 1: TRAIN TOP LAYERS")
print("=" * 60)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    callbacks=callbacks,
    verbose=1
)

# =========================
# PHASE 2 — FINE TUNING
# =========================
print("\n" + "=" * 60)
print(">>> PHASE 2: FINE-TUNING BACKBONE")
print("=" * 60)

base_model = model.get_layer("mobilenetv2_1.00_224")
base_model.trainable = True

# Chỉ mở 30 layer cuối
for layer in base_model.layers[:-30]:
    layer.trainable = False

print(f"✅ Unfrozen last 30 layers")
print(f"✅ Trainable params: {sum(tf.size(w).numpy() for w in model.trainable_weights):,}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    callbacks=callbacks,
    verbose=1
)

# =========================
# SAVE FINAL MODEL + LABELS
# =========================
model.save(MODEL_PATH)

with open("models/class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

# =========================
# FINAL REPORT
# =========================
print("\n" + "=" * 60)
print("✅ TRAINING COMPLETED")
print("=" * 60)

final_train_acc = history2.history["accuracy"][-1]
final_val_acc = history2.history["val_accuracy"][-1]

print(f"Final Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")
print(f"Overfitting Gap: {abs(final_train_acc - final_val_acc):.4f}")
print("=" * 60)
