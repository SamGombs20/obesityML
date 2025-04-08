from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from model.input import InputModel, PredictionModel, RecommendationModel
from utils.common import generate_dataframe, preprocess,scaleData,predict as pred, get_recommendation
import joblib
import os

# Get the current directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

#configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Load the model once
try:
    model = joblib.load(os.path.join(MODELS_DIR,"model.pkl"))
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your specific domain for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Obesity Prediction"}
@app.post("/predict", response_model=PredictionModel)
def predict(input:InputModel):
    try:
        logger.info("Received prediction request")
        df = preprocess(generate_dataframe(input))
        scaled_data = scaleData(df)
    
        prediction = model.predict(scaled_data)
        logger.info(f"Prediction result : {prediction[0]}")
        return PredictionModel(prediction=pred(prediction[0]))
    except ValueError as e:
        logger.info(f"Value error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.info(f"Runtime error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.info(f"Unexpected exception: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

@app.post("/recommendation", response_model=RecommendationModel)
def recommendation(input:InputModel):
    try:
        recommendation_text = get_recommendation(input)
        return RecommendationModel(recommendation=recommendation_text)
    except Exception as e:
        logger.error(f"Unexpected exception: {e}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", reload=True)