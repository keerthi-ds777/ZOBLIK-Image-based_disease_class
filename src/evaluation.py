import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, test_dataset):
    # Evaluate the model on test dataset
    test_loss, test_accuracy = model.evaluate(test_dataset)
    
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy:.4f}")
    print(f"Test Accuracy : {test_accuracy * 100:.2f}%")
    
    return test_loss, test_accuracy

def plot_training_history(history):
    # Plot training vs validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot training vs validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

def generate_classification_report_and_matrix(model, test_dataset, class_names):
    y_true = []
    y_pred = []

    # Get predictions for all samples in the test dataset
    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted_classes)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\n====================================")
    print("CLASSIFICATION REPORT")
    print("====================================")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Calculate and plot the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    # Save the final model as specified in the notebook
    model.save("cnn_medical_image_classifier.keras")
    print("\nModel saved as cnn_medical_image_classifier.keras")

    # Save class names JSON
    with open("class_names.json", "w") as f:
        json.dump(class_names, f)
    print("Class names saved as class_names.json")

def predict_image(image_path, model, class_names, image_size=(224, 224)):
    # Load and preprocess the single image
    img = image.load_img(image_path, target_size=image_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Make the prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index]

    print("\n====================================")
    print("IMAGE PREDICTION")
    print("====================================")
    print("Predicted Class:", predicted_class)
    print(f"Confidence: {confidence * 100:.2f}%")

    # Display the image with predictions
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"{predicted_class} ({confidence * 100:.2f}%)")
    plt.axis("off")
    plt.show()

    return predicted_class, confidence
