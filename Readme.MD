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
| XGBoost Detector | Binary — is there an anomaly? | ROC-AUC: 0.9828 |
| XGBoost Classifier | Multi-class — what type? | Accuracy: 0.99 |

### Anomaly Types

| Type | Severity | Description |
|---|---|---|
| fire | Critical | High smoke + temp spike with high variance |
| gas_leak | Critical | Gas rising gradually above baseline |
| water_leak | Medium | water_flow sustained above normal baseline |
| energy_anomaly | Medium | Power spike with high rolling std |
| intrusion | High | Motion detected at night with elevated power |
| sensor_fault | Low | Abnormal readings drop below normal range |

---

## 📁 Project Structure

```
smart-home-ai/
├── models/
│   ├── xgboost_model.pkl          ← Anomaly detector (36 features)
│   ├── xgboost_type_v1.pkl        ← Type classifier (7 features)
│   ├── features.pkl               ← Detector feature list (36)
│   ├── features_type.pkl          ← Classifier feature list (7)
│   └── label_encoder.pkl          ← Anomaly type encoder
├── notebooks/
│   ├── xgboost.ipynb              ← Detector training
│   └── xgboost_classifier.ipynb   ← Classifier training
├── data/
│   └── smart_home_simulated_v7.csv
├── face_recognition/
│   ├── app.py                     ← Flask face recognition service
│   └── requirements.txt
├── main.py                        ← FastAPI anomaly detection
├── test_camera.py                 ← Live camera monitoring script
├── requirements.txt
└── Dockerfile
```

---

## ⚙️ Requirements

- Python 3.11
- MongoDB running locally
- See `requirements.txt` and `face_recognition/requirements.txt`

---

## 🚀 Running Locally

### 1. Install dependencies

```bash
py -3.11 -m pip install -r requirements.txt
```

### 2. Start FastAPI (Anomaly Detection)

```bash
py -3.11 -m uvicorn main:app --reload --port 8000
```

### 3. Start Flask (Face Recognition)

```bash
cd face_recognition
py -3.11 app.py --port 7000 --mongodb-uri mongodb://localhost:27017/ --db-name smarthome
```

---

## 🐳 Running with Docker

```bash
docker build -t smarthome-ai .
docker run -p 8000:8000 smarthome-ai
```

---

## 🌐 Services

| Service | Port | Responsibility |
|---|---|---|
| FastAPI | 8000 | Anomaly detection |
| Flask | 7000 | Face recognition |
| Node.js | 5000 | Backend API |

---

## 📡 API Endpoints

### FastAPI — Anomaly Detection

#### `GET /`
Health check.

**Response:**
```json
{ "status": "ok" }
```

---

#### `POST /predict`
Main prediction endpoint. Called by the Node.js backend.

**Important:** The Node.js backend is responsible for computing all lag and rolling features from the last 5 readings in MongoDB. The frontend only sends the 7 base sensor values to Node, and Node sends the full 36 features to this endpoint.

**Request Body (36 features):**
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

> FastAPI computes internally: `is_night`, `motion_door`, `high_power`, `high_smoke`, `smoke_gas_ratio`, `temp_power_ratio` — Node does NOT need to send these.

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

### Flask — Face Recognition

#### `GET /`
Health check.

#### `POST /face-register`
Register a known face.

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| file | File (jpg/png) | Person's photo |
| name | Text | Person's name — no spaces in key |

**Response:**
```json
{
  "status": "success",
  "message": "Face registered successfully as Ahmed",
  "user_id": "uuid...",
  "name": "Ahmed"
}
```

#### `POST /face-auth`
Verify a face against registered faces.

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| file | File (jpg/png) | Photo to verify |
| threshold | Text (optional) | Confidence threshold, default 0.5 |

**Response:**
```json
{
  "status": "success",
  "faces_count": 1,
  "results": [
    {
      "known": true,
      "name": "Ahmed",
      "user_id": "uuid...",
      "confidence": 0.85
    }
  ]
}
```

---

## 🔗 Integration with Node.js

The Node.js backend is responsible for:

1. Receiving the 7 base sensor values from the frontend
2. Fetching the last 5 readings from MongoDB for the same `deviceId`
3. Computing lag features (`lag1`, `lag2`) and rolling features (`roll_mean`, `roll_std`)
4. Calling `POST /predict` with the full feature set
5. Handling the response — creating anomaly + alert + Socket.io notification if needed
6. For face recognition: forwarding requests to Flask `/face-register` and `/face-auth`

---

## 🧪 Test Examples (via Node.js `POST /api/readings`)

Build history first — send these 5 normal readings sequentially for the same `deviceId`:

```json
{ "deviceId": "<your-device-id>", "temp": 25.1, "smoke": 2.9, "gas": 3.3, "power": 252.3, "motion": 0, "door": 0, "water_flow": 0.0 }
{ "deviceId": "<your-device-id>", "temp": 25.8, "smoke": 2.0, "gas": 3.1, "power": 301.3, "motion": 0, "door": 0, "water_flow": 0.0 }
{ "deviceId": "<your-device-id>", "temp": 24.9, "smoke": 2.4, "gas": 3.5, "power": 248.0, "motion": 0, "door": 0, "water_flow": 0.0 }
{ "deviceId": "<your-device-id>", "temp": 25.2, "smoke": 2.1, "gas": 3.0, "power": 260.5, "motion": 0, "door": 0, "water_flow": 0.0 }
{ "deviceId": "<your-device-id>", "temp": 24.8, "smoke": 1.2, "gas": 2.2, "power": 251.7, "motion": 0, "door": 0, "water_flow": 0.04 }
```

Then send anomaly readings:

**Fire:**
```json
{ "deviceId": "<your-device-id>", "temp": 49.1, "smoke": 82.4, "gas": 4.8, "power": 464.7, "motion": 0, "door": 0, "water_flow": 0.025 }
```

**Gas Leak:**
```json
{ "deviceId": "<your-device-id>", "temp": 22.96, "smoke": 4.38, "gas": 8.15, "power": 348.66, "motion": 0, "door": 0, "water_flow": 0.0 }
```

**Water Leak:**
```json
{ "deviceId": "<your-device-id>", "temp": 22.23, "smoke": 2.51, "gas": 5.46, "power": 251.50, "motion": 0, "door": 0, "water_flow": 2.85 }
```

**Energy Anomaly:**
```json
{ "deviceId": "<your-device-id>", "temp": 23.73, "smoke": 3.07, "gas": 5.11, "power": 1083.70, "motion": 0, "door": 0, "water_flow": 0.007 }
```

**Intrusion:**
```json
{ "deviceId": "<your-device-id>", "temp": 24.41, "smoke": 3.08, "gas": 3.13, "power": 445.20, "motion": 1, "door": 0, "water_flow": 0.008 }
```

**Sensor Fault:**
```json
{ "deviceId": "<your-device-id>", "temp": 23.42, "smoke": 1.79, "gas": 5.42, "power": 251.27, "motion": 0, "door": 0, "water_flow": 0.026 }
```

---

## 👥 Team

| Role | Responsibility |
|---|---|
| AI/ML | Data, models, FastAPI, Flask, Docker |
| Backend | Node.js, MongoDB, Socket.io |
| Frontend | React, Next.js, Dashboard |