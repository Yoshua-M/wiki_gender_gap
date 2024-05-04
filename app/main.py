# main.py
import pickle

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pandas as pd

app = FastAPI()


class Item(BaseModel):
    E_NEds: float = 0.5
    E_Bpag: float = 0.8
    NEds: int = 100
    NDays: int = 10
    NPcreated: int = 1
    ns_user: int = 1
    ns_wikipedia: int = 0
    ns_talk: int = 0
    ns_userTalk: int = 0
    # Add more features as needed



@app.post("/predict")
async def predict(features: Item):
    try:
        # Load the serialized model
        def load_model(filename):
            with open(filename, 'rb') as f:
                model = pickle.load(f)
            return model

        # Deserialize the model
        print('loading model')
        model = load_model("models/model_v1_2024-05-02.pkl")
        preprocessor = load_model("models/preprocessor_2024-05-02.pkl")
        print('model loading complete')

        # Prepare input features for prediction
        print('features:')
        print(features.__dict__)
        features_df = pd.DataFrame(features.__dict__, index=[0])

        # Make predictions
        print('attempting prediction')
        prediction = model.predict(preprocessor.transform(features_df))

        # Return the prediction as JSON response
        return {"predicted class": prediction[0]}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Hello World"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)