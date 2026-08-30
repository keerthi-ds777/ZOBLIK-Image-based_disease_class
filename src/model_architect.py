import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(num_classes, img_height=224, img_width=224, data_augmentation=None):
    

    data_augmentation = tf.keras.Sequential([
    
    layers.RandomFlip("horizontal"),

    layers.RandomRotation(0.05 ),

    layers.RandomZoom(0.10),

    layers.RandomContrast(0.10)

    ], name="data_augmentation")


    
    # model Building

    model = models.Sequential([

    # Input layer
    layers.Input(shape=(img_height, img_width, 3)), 
    data_augmentation,
    
    layers.Rescaling(1.0 / 255),
    
    layers.Conv2D(filters=32,kernel_size=(3, 3),activation="relu",padding="same", kernel_regularizer=l2_reg),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(filters=64,kernel_size=(3, 3),activation="relu",padding="same",kernel_regularizer=l2_reg),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
   
    layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same",kernel_regularizer=l2_reg),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D( filters=256,kernel_size=(3, 3),activation="relu",padding="same", kernel_regularizer=l2_reg),
    layers.BatchNormalization(),
    layers.MaxPooling2D( pool_size=(2, 2)),

    layers.SpatialDropout2D(0.30),

    layers.GlobalAveragePooling2D(),

    layers.Dense(128,activation="relu",kernel_regularizer=l2_reg),
    layers.BatchNormalization(),

    layers.Dropout(0.40),

    layers.Dense(num_classes,activation="softmax")

    ])

    model.summary()



    
    return model
