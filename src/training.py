import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt

def train_model(model, train_dataset, val_dataset, early_stopping, model_checkpoint, reduce_lr):
    print("\nStarting CNN training...\n")
    EPOCHS = 10

    # Disable saving the model artifacts
    mlflow.tensorflow.autolog(log_models=False)

    with mlflow.start_run():
        
        mlflow.set_tag("run_purpose", "Fixing overfitting on CT scans")
        mlflow.set_tag("architecture_changes", "Added L2 reg (1e-4) in all Conv2D and SpatialDropout(0.30)")
        
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            callbacks=[
                early_stopping,
                model_checkpoint,
                reduce_lr
            ]
        )
        
        fig_acc = plt.figure(figsize=(10, 6))
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Training vs Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid()
        
        # Save directly to MLflow (no local file needed)
        mlflow.log_figure(fig_acc, "visualizations/accuracy_curve.png")
        
        # Clear the figure from memory so they don't overlap
        plt.show()
        plt.close(fig_acc)

        fig_loss = plt.figure(figsize=(10, 6))
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        
        # Save the second plot to MLflow
        mlflow.log_figure(fig_loss, "visualizations/loss_curve.png")
        
        # Clear the memory
        plt.show()
        plt.close(fig_loss)
