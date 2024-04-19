"""
All necessary functions for data transformation in the project.
"""

import pandas as pd
from scipy.stats import boxcox

from src.utils import Kernel

def data_selection(df: pd.DataFrame, kernel: Kernel) -> pd.DataFrame:
    config = kernel.config
    df = df[df[config['target']] != 0]
    df = df[config['selected_features']]

    print(f'Gender-defined records: {len(df)}')

    return df

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


# %%
kernel = Kernel()
df = pd.read_csv('src/mlops_data.csv')
df_selected = data_selection(df, kernel)
df_selected = remove_outliers(df_selected, kernel)
df_selected = feature_normalization(df_selected, kernel)




