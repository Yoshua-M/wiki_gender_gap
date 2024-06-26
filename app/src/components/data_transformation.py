"""
All necessary functions for data transformation in the project.
"""

import pandas as pd
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from typing import Tuple
import numpy as np

from src.utils import Kernel


def remove_outliers(df: pd.DataFrame, kernel: Kernel) -> pd.DataFrame:
    config = kernel.config
    numeric_df = df[config['numeric_features']]

    # Calculate the Interquartile Range (IQR) for each numeric feature
    Q1 = numeric_df.quantile(config['interquantile_range'][0])
    Q3 = numeric_df.quantile(config['interquantile_range'][1])
    IQR = Q3 - Q1

    # Identify outliers using the IQR method for the subset of columns
    outliers = ((numeric_df < (Q1 - config['outlier_std'] * IQR)) | (
                numeric_df > (Q3 + config['outlier_std'] * IQR))).sum()

    # Count outliers for each feature
    print("Number of outliers for each feature:")
    print(outliers)

    # Remove outliers from the dataset
    df = df[~((numeric_df < (Q1 - config['outlier_std'] * IQR)) | (
                numeric_df > (Q3 + config['outlier_std'] * IQR))).any(axis=1)]

    # Print the shape of the cleaned dataset
    print("Shape of the cleaned dataset after removing outliers:", df.shape)

    return df


def feature_normalization(df: pd.DataFrame, kernel: Kernel) -> pd.DataFrame:
    config = kernel.config
    df[config['numeric_features']] = df[config['numeric_features']].apply(
        lambda x: boxcox(x + 1)[0])

    return df


kernel = Kernel()
config = kernel.config


class Preprocessor:
    numeric_features = config['numeric_features']
    target = config['target']

    def slice(self, df: pd.DataFrame) -> Tuple:
        """Add splitting and writing: train testing and predict"""
        df_unlabeled = df[df[self.target] == 0]
        df_labeled = df[df[self.target] != 0]
        # print(f'Gender-defined records: {len(df)}')

        return df_unlabeled, df_labeled

    def select_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.numeric_features + [self.target]]

    def normilize(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.numeric_features] = df[self.numeric_features].apply(
            lambda x: np.log(x + 1))
        return df

    def apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_array, index=df.index,
                                 columns=df.columns)
        return scaled_df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.select_data(df)
        labels = df[self.target]
        features = df[self.numeric_features]
        features = self.normilize(features)
        features = self.apply_scaling(features)
        preprocessed = pd.concat([features, labels], axis=1)
        return preprocessed


def apply_bootstrap_balance(df: pd.DataFrame, kernel: Kernel) -> pd.DataFrame:
    # Separate majority (male) and minority (female) classes
    majority_df = df[df['gender'] == 1]
    minority_df = df[df['gender'] == 2]

    # Bootstrap resample the minority class to balance the dataset
    minority_resampled = resample(minority_df, n_samples=len(majority_df),
                                  replace=True, random_state=42)

    # Combine resampled minority class with majority class
    balanced_df = pd.concat([majority_df, minority_resampled])

    # Shuffle the indices to integrate minority samples with majority
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(
        drop=True)

    print("Number of Males in balanced dataset:",
          len(balanced_df[balanced_df['gender'] == 1]))
    print("Number of Females in balanced dataset:",
          len(balanced_df[balanced_df['gender'] == 2]))

    return balanced_df







