"""
Training and assesing a model
"""

import pandas as pd
from src.pipeline import data_pipeline
from sklearn.linear_model import LogisticRegression

from src.utils import Kernel
from src.components import model_trainer

kernel = Kernel()
df = pd.read_csv('src/mlops_data.csv')
df = data_pipeline.run(df, kernel)

model = LogisticRegression(max_iter=1000)
model_trainer.train_and_evaluate(model, df, kernel)