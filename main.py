from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from datetime import datetime
 
app = FastAPI()

# ── Load Models ──
detector = joblib.load("models/xgboost_model.pkl")
classifier = joblib.load("models/xgboost_type_v1.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# ── Request Schema ──
class SensorData(BaseModel):
    deviceId: str
    temp: float
    smoke: float
    gas: float
    power: float
    motion: int
    door: int
    water_flow: float
    temp_lag1: float = 0
    temp_lag2: float = 0
    smoke_lag1: float = 0
    smoke_lag2: float = 0
    gas_lag1: float = 0
    gas_lag2: float = 0
    power_lag1: float = 0
    power_lag2: float = 0
    water_flow_lag1: float = 0
    water_flow_lag2: float = 0
    temp_roll_mean: float = 0
    temp_roll_std: float = 0
    smoke_roll_mean: float = 0
    smoke_roll_std: float = 0
    gas_roll_mean: float = 0
    gas_roll_std: float = 0
    power_roll_mean: float = 0
    power_roll_std: float = 0
    water_flow_roll_mean: float = 0
    water_flow_roll_std: float = 0

# ── Predict Endpoint ──
@app.post("/predict")
def predict(data: SensorData):
    now = datetime.now()

    motion_door = data.motion * data.door
    high_power = 1 if data.power > 400 else 0
    high_smoke = 1 if data.smoke > 10 else 0

    
    features_detector = [
        data.temp, data.smoke, data.gas, data.power,
        data.motion, data.door, data.water_flow,
        now.hour, now.day, now.weekday(),
        data.temp_lag1, data.temp_lag2,
        data.smoke_lag1, data.smoke_lag2,
        data.gas_lag1, data.gas_lag2,
        data.power_lag1, data.power_lag2,
        data.water_flow_lag1, data.water_flow_lag2,
        data.temp_roll_mean, data.temp_roll_std,
        data.smoke_roll_mean, data.smoke_roll_std,
        data.gas_roll_mean, data.gas_roll_std,
        data.power_roll_mean, data.power_roll_std,
        data.water_flow_roll_mean, data.water_flow_roll_std,
        motion_door, high_power, high_smoke,
    ]

    
    features_classifier = [
        data.temp, data.smoke, data.gas, data.power,
        data.motion, data.door, data.water_flow
    ]

    X_detector = np.array(features_detector).reshape(1, -1)
    X_classifier = np.array(features_classifier).reshape(1, -1)

   
    prob = detector.predict_proba(X_detector)[0][1]
    is_anomaly = bool(prob > 0.5)

    if not is_anomaly:
        return {
            "isAnomaly": False,
            "type": None,
            "confidence": round(float(prob), 3)
        }

    
    type_pred = classifier.predict(X_classifier)[0]
    anomaly_type = label_encoder.inverse_transform([type_pred])[0]

    return {
        "isAnomaly": True,
        "type": anomaly_type,
        "confidence": round(float(prob), 3)
    }

# ── Health Check ──
@app.get("/")
def health():
    return {"status": "ok"}