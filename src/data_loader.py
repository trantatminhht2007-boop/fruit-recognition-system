import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

def load_datasets(data_dir, img_size=(224, 224), batch_size=32, seed=42):
    """
    Load và optimize datasets với prefetch, cache, và shuffle.
    
    Args:
        data_dir: Đường dẫn tới thư mục dataset
        img_size: Kích thước ảnh đầu ra (default: 224x224)
        batch_size: Số ảnh mỗi batch (default: 32)
        seed: Random seed cho reproducibility (default: 42)
    
    Returns:
        train_ds, val_ds, class_names
    """
    train_dir = f"{data_dir}/train"
    val_dir = f"{data_dir}/val"
    
    # Load datasets với seed để reproducible
    train_ds = image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True,
        seed=seed
    )
    
    val_ds = image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False
    )
    
    # 👉 Lấy class_names ngay tại đây
    class_names = train_ds.class_names
    
    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_ds = train_ds.cache().shuffle(1000, seed=seed).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    
    print(f"✅ Loaded {len(class_names)} classes: {class_names}")
    print(f"✅ Training batches: {len(train_ds)}")
    print(f"✅ Validation batches: {len(val_ds)}")
    
    return train_ds, val_ds, class_names
