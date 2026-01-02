import tensorflow as tf
from tensorflow.keras import layers, models

def build_model():
    # Base Model (MobileNetV2)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), 
        include_top=False, 
        weights='imagenet'
    )
    base_model._name = "mobilenet_base" # Đặt tên để dễ gọi lại
    base_model.trainable = False 

    inputs = layers.Input(shape=(224, 224, 3))
    # Nhúng Preprocessing vào model để predict.py không bị sai dải màu
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs) 
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x) # Chống overfitting cho tập dữ liệu nhỏ
    outputs = layers.Dense(5, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy', # Khớp với data_loader
        metrics=['accuracy']
    )
    return model