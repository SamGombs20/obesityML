import pandas as pd
from model.input import InputModel
import joblib
import os

# Get the current directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "../models")

#Load the encoder and scaler once
try:
    encoder = joblib.load(os.path.join(MODELS_DIR, "encoder.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
except Exception as e:
    raise RuntimeError(f"Failed to load encoder or scaler: {e}")


def generate_dataframe(data:InputModel):
    return pd.DataFrame([data.model_dump()])
def  preprocess(df:pd.DataFrame)-> pd.DataFrame:
    
    #Binary encoding for binary data
    binary_features = ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
    df[binary_features] = df[binary_features].replace({"yes":1, "no":0})

    #One-Hot encoding for categorical features
    categorical_features = ["Gender", "CAEC", "CALC", "MTRANS"]
    encoded_cats = encoder.transform(df[categorical_features])
    encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_features))

    df = df.drop(columns=df[categorical_features])
    df = pd.concat([df,encoded_cat_df], axis=1)
    
    return df

def predict(prediction:int)-> str:
    prediction_map = {
        0: "Insufficient_Weight",
        1: "Normal_Weight",
        2: "Overweight_Level_I",
        3: "Overweight_Level_II",
        4: "Obesity_Type_I",
        5: "Obesity_Type_II",
        6: "Obesity_Type_III"
    }
    return prediction_map.get(prediction, "Invalid Prediction")

def scaleData(df:pd.DataFrame)->pd.DataFrame:
    
    continuous_features = ["Age", "Height", "Weight", "NCP", "CH2O", "FAF"]
    df1 = df
    df1[continuous_features] = scaler.transform(df1[continuous_features])

    return df1

#Recommendation Logic
def get_recommendation(obesity_level:str)->str:
    recommendations = {
        "Insufficient_Weight": "Increase calorie intake with nutrient-dense foods, focus on protein and healthy fats, and consider strength training.",
        "Normal_Weight": "Maintain a balanced diet and regular physical activity to sustain health.",
        "Overweight_Level_I": "Slight calorie reduction, increase daily activity (e.g., walking, cardio exercises).",
        "Overweight_Level_II": "Adopt a structured meal plan, reduce processed food intake, and increase exercise intensity.",
        "Obesity_Type_I": "Engage in moderate-intensity workouts, track calorie intake, and focus on portion control.",
        "Obesity_Type_II": "Consider a personalized diet plan, increase fiber intake, and engage in regular aerobic exercises.",
        "Obesity_Type_III": "Consult a healthcare provider for a structured weight management plan, focus on gradual lifestyle changes, and possibly consider medical interventions if necessary."
    }
    return recommendations.get(obesity_level, "No recommendation available.")