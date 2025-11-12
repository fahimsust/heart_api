from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np, joblib

app = Flask(__name__)
CORS(app)  # allow calls from your Flutter app during dev

model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

# Keep the EXACT same order used in training (including engineered features)
FEATURES = [
  "age","sex","cp","trestbps","chol","fbs","restecg","thalach",
  "exang","oldpeak","slope","ca","thal",
  "chol_to_age","hr_ratio","bp_ratio","risk_index"
]
@app.route("/", methods=["GET"])
def home():
    return "✅ Heart Disease Prediction API is running! Use POST /predict"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    # expect: {"inputs": {feature_name: value, ...}}
    x = np.array([[data["inputs"][f] for f in FEATURES]], dtype=float)
    x_sc = scaler.transform(x)
    proba = float(model.predict_proba(x_sc)[0,1])
    pred = int(proba >= 0.5)
    return jsonify({"prediction": pred, "probability": proba})

if __name__ == "__main__":
    # 0.0.0.0 so devices on same Wi-Fi can reach it
    app.run(host="0.0.0.0", port=5000, debug=True)
