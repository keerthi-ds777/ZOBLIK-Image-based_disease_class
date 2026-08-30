from src.load_dataset import load_dataset_and_preprocess
from src.model_architect import build_model_architect
from src.model_optimizing import optimize_model
from src.training import train_model
from src.evaluation import evaluate_model

def main():
    #load dataset and preprocessing
    train_dataset, val_dataset, test_dataset, IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES = load_dataset_and_preprocess()

    #model architect
    model = build_model_architect(IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES)

    #model optimizing
    model, early_stopping, model_checkpoint, reduce_lr = optimize_model(model)

    #training
    train_model(model, train_dataset, val_dataset, early_stopping, model_checkpoint, reduce_lr)

    #evaluation
    evaluate_model(model, test_dataset)

if __name__ == "__main__":
    main()
