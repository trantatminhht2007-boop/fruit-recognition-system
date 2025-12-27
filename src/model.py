import tensorflow as tf
from tensorflow.keras import layers, models

def build_model():
    # 1. Lớp Augmentation (Sửa đổi: Thêm RandomContrast để mô phỏng điều kiện ánh sáng thực tế)
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2), # Giúp model chịu được ánh sáng yếu/mạnh
    ], name="data_augmentation")

    # 2. Base Model (MobileNetV2 Frozen)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), 
        include_top=False, 
        weights='imagenet'
    )
    base_model.trainable = False # Giữ nguyên chiến lược đóng băng

    # 3. Lắp ghép Model (Thêm Preprocessing của chính MobileNetV2)
    inputs = layers.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x) # Rất quan trọng để khớp weights chuẩn của Google
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x) # Tăng lên 0.3 để chống học vẹt tốt hơn với data nhỏ
    outputs = layers.Dense(5, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    # 4. Compile với Learning Rate cực nhỏ
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model