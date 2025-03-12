import pandas as pd
from model.input import InputModel
import joblib
import os



def generate_dataframe(data:InputModel):
    return pd.DataFrame([data.model_dump()])
def  preprocess(df:pd.DataFrame)-> pd.DataFrame:
    encoder = joblib.load("../encoder.pkl")
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
    
    scaler = joblib.load("../scaler.pkl")
    continuous_features = ["Age", "Height", "Weight", "NCP", "CH2O", "FAF"]
    df1 = df
    df1[continuous_features] = scaler.transform(df1[continuous_features])

    return df1