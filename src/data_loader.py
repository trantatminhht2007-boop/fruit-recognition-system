import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

def load_datasets(data_dir, img_size=(224, 224), batch_size=32):
    train_dir = f"{data_dir}/train"
    val_dir = f"{data_dir}/val"
    
    train_ds = image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical' # Trả về vector [0, 1, 0, 0, 0]
    )
    
    val_ds = image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    
    return train_ds, val_ds, train_ds.class_names