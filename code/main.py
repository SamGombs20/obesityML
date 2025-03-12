from fastapi import FastAPI
import os
from model.input import InputModel
from utils.common import generate_dataframe, preprocess,scaleData,predict as pred
import joblib

app = FastAPI()

model_path = os.path.abspath("../data/rf_model.pkl")

model = joblib.load("../data/rf_model.pkl")
@app.get("/")
def root():
    return {"message": "Obesity Prediction"}
@app.post("/predict")
def predict(input:InputModel):
    df = preprocess(generate_dataframe(input))
    scaled_data = scaleData(df)
    prediction = model.predict(scaled_data)
    
    return {"Prediction":pred(prediction[0])}
    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)