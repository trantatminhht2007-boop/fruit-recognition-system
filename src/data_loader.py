import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Giữ nguyên các hằng số này để dễ quản lý
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def load_datasets(dataset_dir): # Đổi tên hàm thành load_datasets cho giống mentor yêu cầu
    train_ds = image_dataset_from_directory(
        dataset_dir + "/train",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    val_ds = image_dataset_from_directory(
        dataset_dir + "/val",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    # Lấy danh sách tên class (apple, banana...) từ dataset
    class_names = train_ds.class_names
    
    return train_ds, val_ds, class_names