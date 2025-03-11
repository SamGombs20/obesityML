from fastapi import FastAPI
from model.input import Input

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Obesity Prediction"}
@app.post("/predict")
async def predict(input:Input):
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)