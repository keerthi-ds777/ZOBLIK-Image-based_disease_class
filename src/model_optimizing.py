import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def compile_model(model, learning_rate=0.001):
    # Compile the model with Adam optimizer and sparse categorical crossentropy loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def get_callbacks(checkpoint_name="best_cnn_model.keras"):
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1
    )

    # Model checkpoint callback to save the best model weights
    model_checkpoint = ModelCheckpoint(
        checkpoint_name,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    # Learning rate reduction callback
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )

    return [early_stopping, model_checkpoint, reduce_lr]
