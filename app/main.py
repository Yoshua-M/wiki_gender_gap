# main.py
import pickle

from fastapi import FastAPI, Response, UploadFile, File
from io import StringIO
from pydantic import BaseModel
import uvicorn
import pandas as pd

app = FastAPI()


# Function to load the model and preprocessor
def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# Load model and preprocessor
model = load_model("models/model.pkl")
preprocessor = load_model("models/preprocessor.pkl")


@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    # Read the uploaded file
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode("utf-8")))

    # Preprocess the data
    df_unlabeled, df_labeled = preprocessor.slice(df)

    unlabeled_preprocessed = preprocessor.preprocess(df_unlabeled)
    labaled_preprocessed = preprocessor.preprocess(df_labeled)
    test_labels = labaled_preprocessed[preprocessor.target]
    test_features = labaled_preprocessed.drop(preprocessor.target, axis=1)
    features = unlabeled_preprocessed.drop(preprocessor.target, axis=1)

    # Make batch predictions
    client_predictions = model.predict(features)
    monitor_predictions = model.predict(test_features)

    # Add predictions to the dataframe
    df_unlabeled['predicted_class'] = client_predictions
    df_labeled['predicted_class'] = monitor_predictions

    score = model.score(test_features, test_labels)
    if score < 0.8:
        print("You're model performance might be drifting or need "
              "rearrangements")
        print(f"Model's score (recall): {score}")
    else:
        print(f"Model's score (recall): {score}")

    # Convert dataframe to CSV
    output = StringIO()
    df_unlabeled.to_csv(output, index=False)
    output.seek(0)

    # Return the predictions as a CSV file
    return Response(content=output.getvalue(), media_type="text/csv")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
