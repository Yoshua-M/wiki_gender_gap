"""
Data preparation for training and scoring
"""
import pandas as pd

from src.utils import Kernel
from src.components.data_transformation import *


def run(data: pd.DataFrame, kernel: Kernel) -> pd.DataFrame:
    df = data_selection(data, kernel)
    df = remove_outliers(df, kernel)
    df = feature_normalization(df, kernel)
    df = apply_bootstrap_balance(df, kernel)

    return df
