import os
import tensorflow as tf

def load_dataset_and_preprocess():
    TRAIN_DIR = "/kaggle/input/datasets/edoardovantaggiato/covid19-xray-two-proposed-databases/Datasets/3-classes/Train"
    VAL_DIR = "/kaggle/input/datasets/edoardovantaggiato/covid19-xray-two-proposed-databases/Datasets/3-classes/Val"
    TEST_DIR = "/kaggle/input/datasets/edoardovantaggiato/covid19-xray-two-proposed-databases/Datasets/3-classes/Test"

    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)
    BATCH_SIZE = 32
    SEED = 42

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = train_dataset.class_names
    NUM_CLASSES = len(class_names)

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(
        buffer_size=AUTOTUNE
    )

    val_dataset = val_dataset.prefetch(
        buffer_size=AUTOTUNE
    )

    test_dataset = test_dataset.prefetch(
        buffer_size=AUTOTUNE
    )

    return train_dataset, val_dataset, test_dataset, IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES
