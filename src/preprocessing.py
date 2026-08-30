import tensorflow as tf
from tensorflow.keras import layers

def get_data_augmentation():
    # Define data augmentation sequential layer
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10)
    ], name="data_augmentation")
    
    return data_augmentation

def prepare_dataset(dataset):
    # Prefetch the dataset to optimize pipeline performance
    AUTOTUNE = tf.data.AUTOTUNE
    return dataset.prefetch(buffer_size=AUTOTUNE)
