from pydantic import BaseModel, Field
from typing import Literal

class InputModel(BaseModel):
    Gender: Literal["Male", "Female"]
    Age: int = Field(...,gt=0, description="Age must be greater than 0")
    Height: float = Field(...,gt=0,description="Height must be greater than 0")
    Weight: float = Field(..., gt=0, description="Weight must be greater than 0")
    family_history_with_overweight:Literal["yes", "no"]
    FAVC: Literal["yes", "no"]
    FCVC: float
    NCP: float = Field(..., ge=0, description="Number of main meals per day")
    CAEC: Literal["no", "Sometimes", "Frequently", "Always"]
    SMOKE: Literal["yes", "no"]
    CH2O: float = Field(..., ge=0, description="Water intake in liters")
    SCC: Literal["yes", "no"]
    FAF: float = Field(..., ge=0, description="Physical activity level")
    TUE: float
    CALC: Literal["no", "Sometimes", "Frequently", "Always"]
    MTRANS: Literal["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"]