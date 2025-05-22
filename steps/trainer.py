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



logger = get_logger(__name__)

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
        model = RandomForestClassifier(random_state=23)
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
        logger.info("metrics computed sucessfully")
    
    except Exception as e:
        logger.error(f"ran into an error {e} while training..")

    return model