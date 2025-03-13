from fastapi import FastAPI, HTTPException
import logging
from model.input import InputModel
from utils.common import generate_dataframe, preprocess,scaleData,predict as pred
import joblib

#configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Load the model once
try:
    model = joblib.load("/home/josh/projects/obesityML/app/model.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")
app = FastAPI()



@app.get("/")
def root():
    return {"message": "Obesity Prediction"}
@app.post("/predict")
def predict(input:InputModel):
    try:
        logger.info("Received prediction request")
        df = preprocess(generate_dataframe(input))
        scaled_data = scaleData(df)
    
        prediction = model.predict(scaled_data)
        logger.info(f"Prediction result : {prediction[0]}")
        return {"Prediction":pred(prediction[0])}
    except ValueError as e:
        logger.info(f"Value error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.info(f"Runtime error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.info(f"Unexpected exception: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)