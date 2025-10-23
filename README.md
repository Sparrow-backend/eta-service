---
title: Delivery ETA Prediction API
emoji: 🚚
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# Delivery ETA Prediction API

This is a Machine Learning powered FastAPI application that predicts delivery times based on:
- Distance (km)
- Courier Experience (years)
- Vehicle Type
- Weather Conditions
- Time of Day
- Traffic Level

## API Endpoints

### Health Check
```
GET /
GET /health
```

### Predict ETA
```
POST /predict
```

**Request Body:**
```json
{
  "Distance_km": 10.5,
  "Courier_Experience_yrs": 2.0,
  "Vehicle_Type": "Scooter",
  "Weather": "Sunny",
  "Time_of_Day": "Morning",
  "Traffic_Level": "Low"
}
```

**Response:**
```json
{
  "predicted_delivery_time": 25.3,
  "input_features": {...},
  "model_version": "1.0.0"
}
```

### Batch Prediction
```
POST /predict/batch
```

### Model Info
```
GET /model/info
```

## Interactive Documentation

Once deployed, visit:
- `/docs` - Swagger UI
- `/redoc` - ReDoc

## Model

This API uses a trained machine learning model with preprocessing pipeline to predict accurate delivery times.
