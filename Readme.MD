# 🏠 Smart Home Anomaly Detection — AI Service

A FastAPI-based AI service that detects anomalies in smart home sensor data using two XGBoost models in a pipeline.

---

## 📌 Overview

This service receives sensor readings from the Node.js backend, computes features, and returns whether an anomaly occurred and what type it is.

```
Node.js → POST /predict → FastAPI → XGBoost Pipeline → Response
```

---

## 🤖 Models

| Model | Task | Performance |
|---|---|---|
| XGBoost Detector | Binary — is there an anomaly? | ROC-AUC: 0.979 |
| XGBoost Classifier | Multi-class — what type? | Accuracy: 0.99 |

### Anomaly Types

| Type | Severity | Description |
|---|---|---|
| fire | Critical | High smoke + temp spike |
| gas_leak | Critical | High gas with gradual rise |
| water_leak | Medium | High water_flow sustained |
| energy_anomaly | Medium | Power spike |
| intrusion | High | Motion + door + elevated power |
| sensor_fault | Low | Abnormal readings drop |

---

## 📁 Project Structure

```
smart-home-ai/
├── models/
│   ├── xgboost_model.pkl          ← Anomaly detector
│   ├── xgboost_type_v1.pkl        ← Type classifier
│   ├── features.pkl               ← Detector feature list (33)
│   ├── features_type.pkl          ← Classifier feature list (7)
│   └── label_encoder.pkl          ← Anomaly type encoder
├── notebooks/
│   ├── xgboost.ipynb              ← Detector training
│   └── xgboost_classifier.ipynb   ← Classifier training
├── data/
│   └── smart_home_simulated_v7.csv
├── main.py                        ← FastAPI app
├── requirements.txt
└── Dockerfile
```

---

## ⚙️ Requirements

- Python 3.11
- See `requirements.txt`

---

## 🚀 Running Locally

### 1. Install dependencies

```bash
py -3.11 -m pip install -r requirements.txt
```

### 2. Start the server

```bash
py -3.11 -m uvicorn main:app --reload --port 8000
```

Server runs at: `http://localhost:8000`

---

## 🐳 Running with Docker

```bash
docker build -t smarthome-ai .
docker run -p 8000:8000 smarthome-ai
```

---

## 📡 API Endpoints

### `GET /`
Health check.

**Response:**
```json
{ "status": "ok" }
```

---

### `POST /predict`
Main prediction endpoint. Called by the Node.js backend.

**Request Body:**
```json
{
  "deviceId": "home-01",
  "temp": 22.0,
  "smoke": 2.0,
  "gas": 5.0,
  "power": 230.0,
  "motion": 0,
  "door": 0,
  "water_flow": 0.01,

  "temp_lag1": 22.0, "temp_lag2": 22.0,
  "smoke_lag1": 2.0, "smoke_lag2": 2.0,
  "gas_lag1": 5.0, "gas_lag2": 5.0,
  "power_lag1": 230.0, "power_lag2": 230.0,
  "water_flow_lag1": 0.01, "water_flow_lag2": 0.01,

  "temp_roll_mean": 22.0, "temp_roll_std": 0.1,
  "smoke_roll_mean": 2.0, "smoke_roll_std": 0.1,
  "gas_roll_mean": 5.0, "gas_roll_std": 0.1,
  "power_roll_mean": 230.0, "power_roll_std": 1.0,
  "water_flow_roll_mean": 0.01, "water_flow_roll_std": 0.0
}
```

> **Note:** The Node.js backend computes all lag and rolling features from the last 5 readings in MongoDB before calling this endpoint. The frontend only sends the 7 base sensor values.

**Response — Normal:**
```json
{
  "isAnomaly": false,
  "type": null,
  "confidence": 0.12
}
```

**Response — Anomaly:**
```json
{
  "isAnomaly": true,
  "type": "fire",
  "confidence": 0.857
}
```

---

## 🧪 Test Examples

### Fire
```json
{
  "deviceId": "home-01",
  "temp": 51.0, "smoke": 96.0, "gas": 47.0, "power": 1400.0,
  "motion": 0, "door": 0, "water_flow": 0.0,
  "temp_lag1": 22.0, "temp_lag2": 22.0,
  "smoke_lag1": 3.0, "smoke_lag2": 3.0,
  "gas_lag1": 5.0, "gas_lag2": 5.0,
  "power_lag1": 230.0, "power_lag2": 230.0,
  "water_flow_lag1": 0.0, "water_flow_lag2": 0.0,
  "temp_roll_mean": 22.0, "temp_roll_std": 0.1,
  "smoke_roll_mean": 3.0, "smoke_roll_std": 0.1,
  "gas_roll_mean": 5.0, "gas_roll_std": 0.1,
  "power_roll_mean": 230.0, "power_roll_std": 1.0,
  "water_flow_roll_mean": 0.0, "water_flow_roll_std": 0.0
}
```

### Gas Leak
```json
{
  "deviceId": "home-01",
  "temp": 22.0, "smoke": 2.0, "gas": 47.0, "power": 230.0,
  "motion": 0, "door": 0, "water_flow": 0.0,
  "temp_lag1": 22.0, "temp_lag2": 22.0,
  "smoke_lag1": 2.0, "smoke_lag2": 2.0,
  "gas_lag1": 8.0, "gas_lag2": 9.0,
  "power_lag1": 230.0, "power_lag2": 230.0,
  "water_flow_lag1": 0.0, "water_flow_lag2": 0.0,
  "temp_roll_mean": 22.0, "temp_roll_std": 0.1,
  "smoke_roll_mean": 2.0, "smoke_roll_std": 0.1,
  "gas_roll_mean": 7.0, "gas_roll_std": 2.0,
  "power_roll_mean": 230.0, "power_roll_std": 1.0,
  "water_flow_roll_mean": 0.0, "water_flow_roll_std": 0.0
}
```

### Water Leak
```json
{
  "deviceId": "home-01",
  "temp": 22.0, "smoke": 2.0, "gas": 4.0, "power": 230.0,
  "motion": 0, "door": 0, "water_flow": 9.1,
  "temp_lag1": 22.0, "temp_lag2": 22.0,
  "smoke_lag1": 2.0, "smoke_lag2": 2.0,
  "gas_lag1": 4.0, "gas_lag2": 4.0,
  "power_lag1": 230.0, "power_lag2": 230.0,
  "water_flow_lag1": 2.0, "water_flow_lag2": 2.0,
  "temp_roll_mean": 22.0, "temp_roll_std": 0.1,
  "smoke_roll_mean": 2.0, "smoke_roll_std": 0.1,
  "gas_roll_mean": 4.0, "gas_roll_std": 0.1,
  "power_roll_mean": 230.0, "power_roll_std": 1.0,
  "water_flow_roll_mean": 3.0, "water_flow_roll_std": 3.0
}
```

### Energy Anomaly
```json
{
  "deviceId": "home-01",
  "temp": 22.0, "smoke": 2.0, "gas": 4.0, "power": 1400.0,
  "motion": 0, "door": 0, "water_flow": 0.0,
  "temp_lag1": 22.0, "temp_lag2": 22.0,
  "smoke_lag1": 2.0, "smoke_lag2": 2.0,
  "gas_lag1": 4.0, "gas_lag2": 4.0,
  "power_lag1": 230.0, "power_lag2": 230.0,
  "water_flow_lag1": 0.0, "water_flow_lag2": 0.0,
  "temp_roll_mean": 22.0, "temp_roll_std": 0.1,
  "smoke_roll_mean": 2.0, "smoke_roll_std": 0.1,
  "gas_roll_mean": 4.0, "gas_roll_std": 0.1,
  "power_roll_mean": 230.0, "power_roll_std": 1.0,
  "water_flow_roll_mean": 0.0, "water_flow_roll_std": 0.0
}
```

### Intrusion
```json
{
  "deviceId": "home-01",
  "temp": 22.0, "smoke": 2.0, "gas": 4.0, "power": 350.0,
  "motion": 1, "door": 1, "water_flow": 0.0,
  "temp_lag1": 22.0, "temp_lag2": 22.0,
  "smoke_lag1": 2.0, "smoke_lag2": 2.0,
  "gas_lag1": 4.0, "gas_lag2": 4.0,
  "power_lag1": 350.0, "power_lag2": 350.0,
  "water_flow_lag1": 0.0, "water_flow_lag2": 0.0,
  "temp_roll_mean": 22.0, "temp_roll_std": 0.1,
  "smoke_roll_mean": 2.0, "smoke_roll_std": 0.1,
  "gas_roll_mean": 4.0, "gas_roll_std": 0.1,
  "power_roll_mean": 350.0, "power_roll_std": 1.0,
  "water_flow_roll_mean": 0.0, "water_flow_roll_std": 0.0
}
```

### Sensor Fault
```json
{
  "deviceId": "home-01",
  "temp": 13.0, "smoke": 0.0, "gas": 0.0, "power": 98.0,
  "motion": 0, "door": 0, "water_flow": 0.0,
  "temp_lag1": 22.0, "temp_lag2": 22.0,
  "smoke_lag1": 2.0, "smoke_lag2": 2.0,
  "gas_lag1": 4.0, "gas_lag2": 4.0,
  "power_lag1": 230.0, "power_lag2": 230.0,
  "water_flow_lag1": 0.0, "water_flow_lag2": 0.0,
  "temp_roll_mean": 22.0, "temp_roll_std": 0.1,
  "smoke_roll_mean": 2.0, "smoke_roll_std": 0.1,
  "gas_roll_mean": 4.0, "gas_roll_std": 0.1,
  "power_roll_mean": 230.0, "power_roll_std": 1.0,
  "water_flow_roll_mean": 0.0, "water_flow_roll_std": 0.0
}
```

---

## 🔗 Integration with Node.js

The Node.js backend is responsible for:
1. Receiving the 7 base sensor values from the device
2. Fetching the last 5 readings from MongoDB for the same `deviceId`
3. Computing lag and rolling features
4. Calling `POST /predict` with all 33 features
5. Handling the response and creating alerts if needed

---

## 👥 Team

| Role | Responsibility |
|---|---|
| AI/ML | Data, models, FastAPI, Docker |
| Backend | Node.js, MongoDB, Socket.io |
| Frontend | React, Next.js, Dashboard |