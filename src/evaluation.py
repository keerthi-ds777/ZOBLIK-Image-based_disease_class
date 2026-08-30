import mlflow

def evaluate_model(model, test_dataset):
    test_loss, test_accuracy = model.evaluate(test_dataset)
    
    print(f"Test Loss      : {test_loss:.4f}")
    print(f"Test Accuracy  : {test_accuracy:.4f}")
    print(f"Test Accuracy  : {test_accuracy * 100:.2f}%")
    
    # Save the test scores into MLflow as written in the notebook
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metrics("Test Accuracy%", (test_accuracy * 100))
