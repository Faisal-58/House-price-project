from src.data_loader import load_data
from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.model_pipeline import train_and_evaluate

if __name__ == "__main__":
    df = load_data("data/raw/aus.csv")
    
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor)

  



