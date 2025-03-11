from pydantic import BaseModel

class Input(BaseModel):
    gender: str
    age: int
    height: float
    weight: float
    family_history:str
    FAVC: str
    FCVC: float
    NCP: float
    CAEC: str
    SMOKE: str
    CH2O: float
    SCC: str
    FAF: float
    TUE: float
    CALC: float
    MTRANS: str