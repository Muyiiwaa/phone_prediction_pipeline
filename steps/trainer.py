from random import Random
import kagglehub
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from zenml import step
from typing_extensions import Annotated
from typing import Optional, Tuple
from zenml.logger import get_logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import wandb
import joblib
import time


logger = get_logger(__name__)

run = wandb.init(
    project= "Mobile Phone Classification Project",
    name= "Version 1.0.1",
    id = "first_run",
    config= {
        "Author": "Muyiwa",
        "Random State": 23,
        "n_estimators": 4,
        "max_depth": 4})

@step 
def train_base_model(X_train:pd.DataFrame, 
X_test:pd.DataFrame, y_train:np.ndarray, y_test: np.ndarray) -> Annotated[RandomForestClassifier, "Base Model"]:
    """_This function trains a base model with no extra hyperparameters
    except random state, outputs serialized model object and returns 
    two numpy arrays of train prediction and test prediction._

    Args:
        X_train (pd.DataFrame): _The training features._
        X_test (pd.DataFrame): _The testing features._
        y_train (pd.DataFrame): _The target variables for the training set._

    Returns:
        tuple[np.array, np.array]: _the train and test predictions._
    """
    model : Optional[RandomForestClassifier] = None
    try:
        model = RandomForestClassifier(random_state=23, max_depth=4,
                                       n_estimators=4)
        model.fit(X_train, y_train)
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        logger.info("completed training, now evaluating...")
        train_precision = precision_score(y_train, train_preds, average = "weighted")
        test_precision = precision_score(y_test, test_preds, average = "weighted")
        train_recall = recall_score(y_train, train_preds, average = "weighted")
        test_recall = recall_score(y_test, test_preds, average = "weighted")
        train_f1 = f1_score(y_train, train_preds, average = "weighted")
        test_f1 = f1_score(y_test, test_preds, average = "weighted")
        logger.info("metrics computed sucessfully now loggin metrics in wandb")

        # log metrics
        run.log(data = {
            "train precision": train_precision,
            "train recall": train_recall,
            "train f1 score": train_f1,
            "test precision": test_precision,
            "test recall": test_recall,
            "test f1": test_f1
        })
        joblib.dump(model, "src/model.pkl")
        logger.info("Logged metrics to wandb dashboard successfully")

    except Exception as e:
        logger.error(f"ran into an error {e} while training..")

    return model

if __name__ == "__main__":
    train_base_model()