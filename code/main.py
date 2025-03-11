from fastapi import FastAPI
from model.input import Input
from utils.common import generate_dataframe, preprocess,scaleData
import pickle

app = FastAPI()

with open ('/home/josh/projects/obesityML/code/rf_model.pkl','rb') as file:
    model = pickle.load(file)
@app.get("/")
def root():
    return {"message": "Obesity Prediction"}
@app.post("/predict")
def predict(input:Input):
    df = preprocess(generate_dataframe(input))
    scaled_data = scaleData(df)
    prediction = model.predict(scaled_data)

    return {"Prediction":prediction}
    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)