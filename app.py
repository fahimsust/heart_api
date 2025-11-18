from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import traceback

app = Flask(__name__)
CORS(app)  # allow calls from your Flutter app during dev

# Load your new model and scaler
model = joblib.load("heart_model.pkl")  # Updated model name
scaler = joblib.load("scaler.pkl")  # Updated scaler name

# Updated feature order for health survey data - MATCHES YOUR CSV
FEATURES = [
    "Sex", "GeneralHealth", "PhysicalHealthDays", "MentalHealthDays", "SleepHours", "BMI"
]

# Categorical value mappings - MUST MATCH YOUR TRAINING ENCODING
CATEGORICAL_MAPPING = {
    "Sex": {"Female": 0, "Male": 1},
    "GeneralHealth": {"Poor": 0, "Fair": 1, "Good": 2, "Very good": 3, "Excellent": 4},
    "PhysicalActivities": {"No": 0, "Yes": 1},
    "AlcoholDrinkers": {"No": 0, "Yes": 1}
}

@app.route("/", methods=["GET"])
def home():
    return "✅ Health Survey Heart Disease Prediction API is running! Use POST /predict"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Validate input
        if not data or "inputs" not in data:
            return jsonify({"error": "Missing 'inputs' in request body"}), 400
        
        inputs = data["inputs"]
        
        # Debug: log received inputs
        print(f"Received input keys: {list(inputs.keys())}")
        
        # Create feature array in correct order with proper encoding
        feature_values = []
        missing_features = []
        
        for feature in FEATURES:
            if feature in inputs:
                value = inputs[feature]
                
                # Handle categorical feature encoding
                if feature in CATEGORICAL_MAPPING:
                    if isinstance(value, str):
                        # Convert string to numeric value
                        if value in CATEGORICAL_MAPPING[feature]:
                            feature_values.append(CATEGORICAL_MAPPING[feature][value])
                        else:
                            # Use default if value not in mapping
                            feature_values.append(0)
                            print(f"Warning: Unknown value '{value}' for feature '{feature}', using default 0")
                    else:
                        # Already numeric, use as-is
                        feature_values.append(value)
                else:
                    # Numerical feature, ensure it's float
                    feature_values.append(float(value))
            else:
                missing_features.append(feature)
                # Provide default value for missing features
                if feature in ["PhysicalHealthDays", "MentalHealthDays", "SleepHours", "BMI", 
                              "total_health_days", "health_days_ratio", "sleep_deviation"]:
                    feature_values.append(0.0)  # Default numerical value
                else:
                    feature_values.append(0)   # Default categorical value
        
        if missing_features:
            print(f"Missing features filled with defaults: {missing_features}")
        
        print(f"Final feature values: {feature_values}")
        print(f"Feature count: {len(feature_values)}")
        
        x = np.array([feature_values], dtype=float)
        
        # Scale features
        x_sc = scaler.transform(x)
        
        # Make prediction
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
    feature_info = {
        "expected_features": FEATURES,
        "expected_feature_count": len(FEATURES),
        "categorical_mappings": CATEGORICAL_MAPPING,
        "numerical_features": [
            "PhysicalHealthDays", "MentalHealthDays", "SleepHours", "BMI",
            "total_health_days", "health_days_ratio", "sleep_deviation"
        ],
        "categorical_features": {
            "Sex": ["Female", "Male"],
            "GeneralHealth": ["Poor", "Fair", "Good", "Very good", "Excellent"],
            "PhysicalActivities": ["No", "Yes"],
            "AlcoholDrinkers": ["No", "Yes"]
        }
    }
    return jsonify(feature_info)

@app.route("/debug", methods=["GET"])
def debug():
    """Debug endpoint to check model and feature compatibility"""
    try:
        model_features = "Unknown"
        scaler_features = "Unknown"
        
        if hasattr(model, 'n_features_in_'):
            model_features = model.n_features_in_
        elif hasattr(model, 'feature_names_in_'):
            model_features = model.feature_names_in_.tolist()
            
        if hasattr(scaler, 'n_features_in_'):
            scaler_features = scaler.n_features_in_
            
        return jsonify({
            "model_expected_features": model_features,
            "scaler_expected_features": scaler_features,
            "our_feature_count": len(FEATURES),
            "our_features": FEATURES,
            "status": "healthy" if model and scaler else "unhealthy"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    print("🚀 Starting Health Survey Heart Disease Prediction API...")
    print(f"📋 Expected features: {FEATURES}")
    print(f"📋 Feature count: {len(FEATURES)}")
    # 0.0.0.0 so devices on same Wi-Fi can reach it
    app.run(host="0.0.0.0", port=5000, debug=True)