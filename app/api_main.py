from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
# Load model
model = joblib.load("models/xgb_pipeline.pkl")

# Define request schema
class HouseFeatures(BaseModel):
    Suburb: str
    Type: str
    Method: str
    SellerG: str
    CouncilArea: str
    Regionname: str
    Rooms: float
    Distance: float
    Postcode: float
    Bedroom2: float
    Bathroom: float
    Car: float
    Landsize: float
    BuildingArea: float
    Latitude: float
    Longitude: float
    Propertycount: float

app = FastAPI()

@app.get("/")
def home():
    return {"message": "House Price Prediction API is live!"}


@app.post("/predict")
def predict(features: HouseFeatures):
    try:
        data = features.model_dump()
        print("✅ Received Payload:", data)

        input_df = pd.DataFrame([data])
        

        
        

        print("📊 DataFrame Columns:", input_df.columns)
        print("📊 DataFrame Preview:\n", input_df.head())

        prediction = model.predict(input_df)[0]
        
        real_price = np.expm1(prediction)
        return {"predicted_price": round(float(real_price), 2)}

        
    except Exception as e:
        print("❌ Prediction Error:", e)
        return {"error": str(e)}



