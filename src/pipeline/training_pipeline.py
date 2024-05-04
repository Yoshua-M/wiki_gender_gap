"""
Training and assesing a model
"""
import pandas as pd
from src.pipeline import data_pipeline
from sklearn.linear_model import LogisticRegression

from src.utils import Kernel
from src.components import model_trainer
import pickle
from datetime import datetime


def serialize_model(model, filename_prefix="model"):
    # Generate filename with current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    filename = f"models/{filename_prefix}_{current_date}.pkl"

    # Serialize model to pickle file
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model serialized and saved as {filename}")



if __name__ == '__main__':

    kernel = Kernel()
    df = pd.read_csv(kernel.config['path_to_data'])
    df = data_pipeline.run(df, kernel)

    model = LogisticRegression(max_iter=1000)
    model_trainer.train_and_evaluate(model, df, kernel)

    serialize_model(model, filename_prefix="model_v1")
