import pandas as pd
from model.input import Input
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

def generate_dataframe(data:Input):
    return pd.DataFrame.from_dict(data.model_dump())
def  preprocess(df:pd.DataFrame)-> pd.DataFrame:
    df['Gender'] = le.fit_transform(df['Gender'])
    df['family_history'] = le.fit_transform(df['family_history'])
    df['FAVC'] = le.fit_transform(df['FAVC'])
    df['CAEC'] = le.fit_transform(df['CAEC'])
    df['SMOKE'] = le.fit_transform(df['SMOKE'])
    df['SCC']= le.fit_transform(df['SCC'])
    df['CALC']= le.fit_transform(df['CALC'])
    df['MTRANS'] = le.fit_transform(df['MTRANS'])
    return df

def predict(prediction)-> str:
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