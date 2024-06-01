"""
Data preparation for training and scoring
"""
import pickle
from datetime import datetime

import pandas as pd

from src.utils import Kernel
from src.components.data_transformation import *


def serialize_model(model, filename_prefix="model"):
    filename = f"models/{filename_prefix}.pkl"
    filename_app = "app/" + filename

    # Serialize model to pickle file
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

    with open(filename_app, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model serialized and saved as {filename}")

def run(data: pd.DataFrame, kernel: Kernel) -> pd.DataFrame:
    preprocessor = Preprocessor()

    unlabeled, df = preprocessor.slice(data)
    df = preprocessor.select_data(df)
    df = remove_outliers(df, kernel)
    df = apply_bootstrap_balance(df, kernel)
    df = preprocessor.preprocess(df)

    serialize_model(preprocessor, 'preprocessor')

    return df
