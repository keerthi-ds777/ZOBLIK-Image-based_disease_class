import tensorflow as tf
from tensorflow.keras import layers, models

def build_model_architect(IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES):
    # Set up L2 regularization so the model architecture constructs successfully
    l2_reg = tf.keras.regularizers.l2(1e-4)

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10)
    ], name="data_augmentation")

    model = models.Sequential([
        # Input layer
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)), 
        data_augmentation,
        
        layers.Rescaling(1.0 / 255),
        
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
       
        layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.SpatialDropout2D(0.30),

        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation="relu", kernel_regularizer=l2_reg),
        layers.BatchNormalization(),

        layers.Dropout(0.40),

        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.summary()
    return model
