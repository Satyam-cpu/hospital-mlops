import os
import csv
from datetime import datetime

import sklearn
print(f"DEBUG: API is using sklearn version: {sklearn.__version__}")

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# -------------------------
# PATH SETUP (VERY IMPORTANT)
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "inpatient_model.pkl")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "predictions.csv")

os.makedirs(LOG_DIR, exist_ok=True)

# Create log file with header if not exists
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "Provider State",
            "DRG Definition",
            "Total Discharges",
            "Average Covered Charges",
            "prediction"
        ])

# -------------------------
# APP & MODEL
# -------------------------
app = FastAPI(title="Hospital Inpatient Charges Prediction API")

model = joblib.load(MODEL_PATH)

# -------------------------
# INPUT SCHEMA
# -------------------------
class InpatientInput(BaseModel):
    Provider_State: str
    DRG_Definition: str
    Total_Discharges: int
    Average_Covered_Charges: float

# -------------------------
# PREDICT ENDPOINT
# -------------------------
@app.post("/predict")
def predict(data: InpatientInput):

    input_df = pd.DataFrame([{
        "Provider State": data.Provider_State,
        "DRG Definition": data.DRG_Definition,
        "Total Discharges": data.Total_Discharges,
        "Average Covered Charges": data.Average_Covered_Charges
    }])

    prediction = float(model.predict(input_df)[0])

    # LOGGING
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            data.Provider_State,
            data.DRG_Definition,
            data.Total_Discharges,
            data.Average_Covered_Charges,
            prediction
        ])

    print("LOG WRITTEN TO:", LOG_FILE)

    return {"predicted_average_total_payment": round(prediction, 2)}
