import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
# Data loader for fruit classification project
# Load images from dataset directory using TensorFlow
# Dataset structure:
# dataset/train/class_name/
# dataset/val/class_name/
# Resize images to 224x224
# Return train and validation datasets
def load_datasets(data_dir, img_size=(224, 224), batch_size=32):
    train_dir = f"{data_dir}/train"
    val_dir = f"{data_dir}/val"
    
    train_ds = image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    
    val_ds = image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    
    class_names = train_ds.class_names
    
    return train_ds, val_ds, class_names