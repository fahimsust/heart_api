from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import traceback

app = Flask(__name__)
CORS(app)

try:
    model = joblib.load("heart_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("✅ Model and scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model/scaler: {e}")
    model = None
    scaler = None

# UPDATED: Exact 6 features your model was trained on
FEATURES = [
    "PhysicalHealthDays", "MentalHealthDays", "SleepHours", 
    "HeightInMeters", "WeightInKilograms", "BMI"
]

@app.route("/", methods=["GET"])
def home():
    return "✅ Health Survey Heart Disease Prediction API is running! Use POST /predict"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({"error": "Model or scaler not loaded"}), 500
        
        data = request.get_json()
        
        if not data or "inputs" not in data:
            return jsonify({"error": "Missing 'inputs' in request body"}), 400
        
        inputs = data["inputs"]
        
        print(f"Received input keys: {list(inputs.keys())}")
        
        # Create feature array in correct order
        feature_values = []
        missing_features = []
        
        for feature in FEATURES:
            if feature in inputs:
                feature_values.append(float(inputs[feature]))
            else:
                missing_features.append(feature)
                feature_values.append(0.0)  # Default numerical value
        
        if missing_features:
            print(f"Missing features filled with defaults: {missing_features}")
        
        print(f"Final feature values: {feature_values}")
        print(f"Feature count: {len(feature_values)}")
        
        x = np.array([feature_values], dtype=float)
        x_sc = scaler.transform(x)
        
        proba = float(model.predict_proba(x_sc)[0, 1])
        pred = int(proba >= 0.5)
        
        return jsonify({
            "prediction": pred, 
            "probability": proba,
            "message": "High risk of heart disease" if pred == 1 else "Low risk of heart disease",
            "confidence": f"{(proba * 100):.1f}%"
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/features", methods=["GET"])
def get_features():
    """Endpoint to see expected features and their types"""
    return jsonify({
        "expected_features": FEATURES,
        "expected_feature_count": len(FEATURES),
        "note": "All features are numerical"
    })

if __name__ == "__main__":
    print("🚀 Starting Health Survey Heart Disease Prediction API...")
    print(f"📋 Expected features: {FEATURES}")
    print(f"📋 Feature count: {len(FEATURES)}")
    app.run(host="0.0.0.0", port=5000, debug=True)