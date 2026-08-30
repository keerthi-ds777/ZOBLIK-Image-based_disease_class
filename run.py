from src.load_dataset import load_data
from src.preprocessing import get_data_augmentation, prepare_dataset
from src.model_architect import build_model
from src.model_optimizing import compile_model, get_callbacks
from src.training import train_model
from src.evaluation import (
    evaluate_model,
    plot_training_history,
    generate_classification_report_and_matrix
)

def main():
    # Step 1: Load Dataset
    train_dataset, val_dataset, test_dataset = load_data()
    
    class_names = train_dataset.class_names
    num_classes = len(class_names)
    print("Classes found:", class_names)
    print("Number of classes:", num_classes)

    # Step 2: Preprocessing
    data_augmentation = get_data_augmentation()
    train_dataset = prepare_dataset(train_dataset)
    val_dataset = prepare_dataset(val_dataset)
    test_dataset = prepare_dataset(test_dataset)

    # Step 3: Model Architect
    model = build_model(
        num_classes=num_classes,
        img_height=224,
        img_width=224,
        data_augmentation=data_augmentation
    )

    # Step 4: Model Optimizing
    model = compile_model(model, learning_rate=0.001)
    callbacks = get_callbacks(checkpoint_name="best_cnn_model.keras")

    # Step 5: Training
    history = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        callbacks=callbacks,
        epochs=10
    )

    # Step 6: Evaluation
    # Plot accuracy and loss curves
    plot_training_history(history)
    
    # Evaluate model on test dataset
    evaluate_model(model, test_dataset)
    
    # Generate classification report and confusion matrix, save final model
    generate_classification_report_and_matrix(model, test_dataset, class_names)

if __name__ == "__main__":
    main()
