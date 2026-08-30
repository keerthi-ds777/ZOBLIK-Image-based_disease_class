import os
import tensorflow as tf

def load_data(train_dir=None, val_dir=None, test_dir=None, img_height=224, img_width=224, batch_size=32, seed=42):
    # Default paths from the notebook
    if train_dir is None:
        train_dir = "/kaggle/input/datasets/edoardovantaggiato/covid19-xray-two-proposed-databases/Datasets/3-classes/Train"
    if val_dir is None:
        val_dir = "/kaggle/input/datasets/edoardovantaggiato/covid19-xray-two-proposed-databases/Datasets/3-classes/Val"
    if test_dir is None:
        test_dir = "/kaggle/input/datasets/edoardovantaggiato/covid19-xray-two-proposed-databases/Datasets/3-classes/Test"

    print("Checking dataset...")
    print("Train path exists:", os.path.exists(train_dir))
    print("Validation path exists:", os.path.exists(val_dir))
    print("Test path exists:", os.path.exists(test_dir))

    image_size = (img_height, img_width)


    print("Loading train dataset...")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )

    print("Loading validation dataset...")
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )

    print("Loading test dataset...")
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataset, val_dataset, test_dataset
