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
def get_recommendation(input_data:InputModel)->str:
    recommendations =[]
    #Weight Analysis (BMI Calculation)
    bmi = input_data.Weight/(input_data.Height **2)
    if bmi<18.5:
        recommendations.append("Increase your calorie intake with nutrient-rich foods like avocados, nuts, and lean proteins.")
    elif bmi>=25:
        recommendations.append("Focus on a balanced diet with portion control and reduce high-calorie foods.")
    
    #Eating Habits
    if input_data.FAVC =="yes":
        recommendations.append("Try to limit frequent consumption of high-calorie foods and opt for healthier alternatives.")
    if input_data.FAVC<2.5:
        recommendations.append("Increase your vegetable consumption for essential nutrients and fiber.")
    if input_data.NCP < 3:
        recommendations.append("Consider eating small, balanced meals throughout the day to maintain energy levels.")
    # Hydration
    if input_data.CH2O < 2:
        recommendations.append("Drink at least 2 liters of water daily to stay hydrated and support metabolism.")

    # Exercise
    if input_data.FAF == 0:
        recommendations.append("Increase physical activity with at least 30 minutes of exercise per day.")
    elif input_data.FAF < 2:
        recommendations.append("Try to add more movement into your daily routine, such as taking the stairs or walking more.")

    # Alcohol & Smoking
    if input_data.CALC in ["Frequently", "Always"]:
        recommendations.append("Consider reducing alcohol consumption as it can contribute to weight gain.")
    if input_data.SMOKE == "yes":
        recommendations.append("Avoid smoking for better overall health and improved metabolism.")

    return " ".join(recommendations) if recommendations else "Maintain your current healthy habits!"

