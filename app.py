from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # allow calls from your Flutter app during dev

# Load your new model and scaler
model = joblib.load("heart_model.pkl")  # Updated model name
scaler = joblib.load("scaler.pkl")  # Updated scaler name

# Updated feature order for health survey data
FEATURES = [
    # Basic demographic
    "age", "sex",
    
    # Health status
    "general_health", "physical_health_days", "mental_health_days", "sleep_hours",
    
    # Physical metrics
    "bmi", "weight_kg", "height_meters",
    
    # Lifestyle factors
    "physical_activities", "alcohol_drinkers",
    
    # Engineered features
    "total_health_days", "health_days_ratio", "bmi_manual", "sleep_deviation", "composite_risk_score"
]

@app.route("/", methods=["GET"])
def home():
    return "✅ Health Survey Heart Disease Prediction API is running! Use POST /predict"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Validate input
        if "inputs" not in data:
            return jsonify({"error": "Missing 'inputs' in request body"}), 400
        
        inputs = data["inputs"]
        
        # Create feature array in correct order
        feature_values = []
        for feature in FEATURES:
            if feature in inputs:
                feature_values.append(inputs[feature])
            else:
                # Provide default value for missing features
                if feature in ["age", "bmi", "weight_kg", "height_meters"]:
                    feature_values.append(0.0)  # Default numerical value
                else:
                    feature_values.append(0)   # Default categorical value
        
        x = np.array([feature_values], dtype=float)
        
        # Scale features
        x_sc = scaler.transform(x)
        
        # Make prediction
        proba = float(model.predict_proba(x_sc)[0, 1])
        pred = int(proba >= 0.5)
        
        return jsonify({
            "prediction": pred, 
            "probability": proba,
            "message": "High risk of heart disease" if pred == 1 else "Low risk of heart disease"
        })
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/features", methods=["GET"])
def get_features():
    """Endpoint to see expected features and their types"""
    feature_info = {
        "numerical_features": [
            "age", "physical_health_days", "mental_health_days", "sleep_hours",
            "bmi", "weight_kg", "height_meters", "total_health_days",
            "health_days_ratio", "bmi_manual", "sleep_deviation", "composite_risk_score"
        ],
        "categorical_features": {
            "sex": [0, 1],  # 0=Female, 1=Male
            "general_health": [0, 1, 2, 3, 4],  # 0=Poor, 1=Fair, 2=Good, 3=Very Good, 4=Excellent
            "physical_activities": [0, 1],  # 0=No, 1=Yes
            "alcohol_drinkers": [0, 1]  # 0=No, 1=Yes
        }
    }
    return jsonify(feature_info)

if __name__ == "__main__":
    # 0.0.0.0 so devices on same Wi-Fi can reach it
    app.run(host="0.0.0.0", port=5000, debug=True)


    curl -X POST https://heart-api-xb4s.onrender.com/predict -H "Content-Type: application/json" -d '{"inputs": {"age":55, "sex":1, "general_health":2, "physical_health_days":2, "mental_health_days":3, "sleep_hours":7, "bmi":25, "weight_kg":70, "height_meters":1.7, "physical_activities":1, "alcohol_drinkers":0}}'