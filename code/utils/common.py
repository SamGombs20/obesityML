import pandas as pd
from model.input import Input

def generate_dataframe(data:Input):
    return pd.DataFrame.from_dict(data.model_dump())
def  preprocess