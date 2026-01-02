import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_model(num_classes=5):
    """
    Build MobileNetV2 model với preprocessing nhúng và L2 regularization.
    
    Args:
        num_classes: Số lượng classes (default: 5)
    
    Returns:
        Compiled Keras model
    """
    # Base Model (MobileNetV2)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model._name = "mobilenet_base"  # Tên consistent để fine-tune
    base_model.trainable = False
    
    inputs = layers.Input(shape=(224, 224, 3), name="input_image")
    
    # Nhúng preprocessing vào model → đảm bảo dải [-1, 1] cho MobileNetV2
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    
    # Thêm L2 regularization để giảm overfitting
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        kernel_regularizer=regularizers.l2(1e-5),
        name="classifier"
    )(x)
    
    model = models.Model(inputs, outputs, name="fruit_classifier")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Model built with {num_classes} classes")
    print(f"✅ Total params: {model.count_params():,}")
    
    return model