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



logger = get_logger(__name__)

# function for preprocessing
@step
def preprocess_data(train_df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "y_test"]]:
    """_This function takes the training data, splits it into
    X and y in terms of features. It also splits the data into
    train set and test set and finally applies standard scaler
    on the all of the predictors (Xs)._

    Args:
        train_df (pd.DataFrame): _The training data._

    Returns:
        tuple[ pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]: _Tuple containing the splits_
    """
    X_train : Optional[pd.DataFrame] = None
    X_test : Optional[pd.DataFrame] = None
    y_train : Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    try:
        X = train_df.drop(columns = ["price_range"])
        y = train_df["price_range"]

        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, stratify = y,random_state=23)

        # scale the datasets
        scaler = StandardScaler()
        scaler.fit(X_train)
        column_names = list(X_train.columns)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # convert X_train and X_test back to a dataframe
        X_train = pd.DataFrame(X_train, columns = column_names)
        X_test = pd.DataFrame(X_test, columns = column_names)
        logger.info(msg= f"Completed Splitting and scaling successfully!")
    except Exception as e:
        logger.error(f"Encountered error {e} while preprocessing!")

    return X_train, X_test, y_train.values, y_test.values