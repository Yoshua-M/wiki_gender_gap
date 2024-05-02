"""
Training and assesing a model
"""
import pandas as pd
from src.pipeline import data_pipeline
from sklearn.linear_model import LogisticRegression

from src.utils import Kernel
from src.components import model_trainer

if __name__ == '__main__':

    kernel = Kernel()
    df = pd.read_csv(kernel.config['path_to_data'])
    df = data_pipeline.run(df, kernel)

    model = LogisticRegression(max_iter=1000)
    model_trainer.train_and_evaluate(model, df, kernel)
