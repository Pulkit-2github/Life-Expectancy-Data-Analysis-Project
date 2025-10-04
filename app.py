# app.py
import os
import pickle
import traceback
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "life_expectancy_model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept JSON with either:
      - a single record as dict: {"feature1": val1, "feature2": val2, ...}
      - a list of records: [{"f1":v1,...}, {...}, ...]
    Returns: {"predictions": [..]}
    """
    try:
        data = request.get_json(force=True)
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        preds = model.predict(df)
        return jsonify({"predictions": [float(p) for p in preds]})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
