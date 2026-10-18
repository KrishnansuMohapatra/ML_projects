import io

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

app = FastAPI(title="House Price Prediction API")

# Fix: Load the model from house_model.joblib
model = joblib.load("house_model.joblib")
features = joblib.load("house_features.joblib")

class HouseFeatures(BaseModel):
    MedInc: float = Field(gt=0, description="Median income in block group")
    HouseAge: float = Field(gt=0, description="Median house age in block group")
    AveRooms: float = Field(gt=0, description="Average number of rooms per household")
    AveBedrms: float = Field(gt=0, description="Average number of bedrooms per household")
    Population: float = Field(gt=0, description="Block group population")
    AveOccup: float = Field(gt=0, description="Average number of household members")
    Latitude: float = Field(description="Block group latitude")
    Longitude: float = Field(description="Block group longitude")

@app.get("/")
def home():
    return {"message": "Welcome to the House Price Prediction API"}

@app.post("/predict")
def predict(data: HouseFeatures):
    try:
        # Create a dictionary from the input data
        input_data = data.model_dump()
        
        # Convert to pandas DataFrame with the correct feature names
        df = pd.DataFrame([input_data])
        
        # Ensure column order matches the model's training features
        df = df[features]
        
        # Make prediction
        prediction = model.predict(df)
        
        # Return prediction (California housing target is in 100,000s)
        return {"predicted_price": float(prediction[0])}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/excel")
async def predict_excel(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Please upload an .xlsx or .xls file")

    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read Excel file: {e}")

    missing = [col for col in features if col not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required columns: {missing}")

    try:
        predictions = model.predict(df[features])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    df["predicted_price"] = predictions * 100000

    output = io.BytesIO()
    df.to_excel(output, index=False, engine="openpyxl")
    output.seek(0)

    headers = {"Content-Disposition": 'attachment; filename="predictions.xlsx"'}
    media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return StreamingResponse(output, media_type=media_type, headers=headers)


def main():
    print("Starting the FastAPI server on http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
