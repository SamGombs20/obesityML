from pydantic import BaseModel

class InputModel(BaseModel):
    Gender: str
    Age: int
    Height: float
    Weight: float
    family_history_with_overweight:str
    FAVC: str
    FCVC: float
    NCP: float
    CAEC: str
    SMOKE: str
    CH2O: float
    SCC: str
    FAF: float
    TUE: float
    CALC: str
    MTRANS: str