import kagglehub
import os
import pandas as pd
import numpy as np
from typing_extensions import Annotated
from zenml import step
from typing import Optional, Tuple
from zenml.logger import get_logger

# configure the logger
logger = get_logger(__name__)

# define the load data function.
@step
def load_data() -> Annotated[pd.DataFrame, "Training Data"]:
    """_This function loads the mobile phone dataset directly
    from kaggle using kagglehub. The data has already splited into
    train and test set from the source._

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: _The train set and the test set._
    """
    train_df: Optional[pd.DataFrame] = None
    try:
        path = kagglehub.dataset_download("iabhishekofficial/mobile-price-classification")
        train_csv_url = os.path.join(path, "train.csv")
        train_df = pd.read_csv(train_csv_url)
        logger.info(msg = f"""
        Data Loaded succesfully with the following info:
        data_shape: {train_df.shape}
        categorical_columns: {list(train_df.select_dtypes(include = "object"))}
        continous_columns: {list(train_df.select_dtypes(exclude = "object"))}
        """)
    except Exception as e:
        logger.error(msg= f"Ecountered error {e} while loading the data")

    return train_df

if __name__ == "__main__":
    data = load_data()
    print(data.columns)
    