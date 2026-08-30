import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt

def train_model(model, train_dataset, val_dataset, callbacks, epochs=10):
    print("\nStarting CNN training...\n")

    # Disable saving the model artifacts in autologging as per the notebook
    mlflow.tensorflow.autolog(log_models=False)

    with mlflow.start_run():
        # Set tags from the notebook
        mlflow.set_tag("run_purpose", "Fixing overfitting on CT scans")
        mlflow.set_tag("architecture_changes", "Added L2 reg (1e-4) in all Conv2D and SpatialDropout(0.30)")

        # Fit the model
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks
        )

        # Plot and log accuracy curve to MLflow
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
        plt.show()
        plt.close(fig_acc)

        # Plot and log loss curve to MLflow
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
        plt.show()
        plt.close(fig_loss)

    return history
