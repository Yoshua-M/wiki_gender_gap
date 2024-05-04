"""
Data preparation for training and scoring
"""
import pickle
from datetime import datetime

import pandas as pd

from src.utils import Kernel
from src.components.data_transformation import *


def serialize_model(model, filename_prefix="model"):
    # Generate filename with current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    filename = f"models/{filename_prefix}_{current_date}.pkl"

    # Serialize model to pickle file
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model serialized and saved as {filename}")

def run(data: pd.DataFrame, kernel: Kernel) -> pd.DataFrame:
    df = data_selection(data, kernel)
    df = remove_outliers(df, kernel)
    df = feature_normalization(df, kernel)
    df = apply_bootstrap_balance(df, kernel)

    preprocessor = Preprocessor()
    serialize_model(preprocessor, 'preprocessor')

    return df
