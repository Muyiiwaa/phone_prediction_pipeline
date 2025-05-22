from zenml import pipeline
from steps.data_loader import  load_data
from steps.preprocess import preprocess_data
from steps.trainer import train_base_model


@pipeline
def run_phone_pipeline():
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)
    train_base_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    run_phone_pipeline()