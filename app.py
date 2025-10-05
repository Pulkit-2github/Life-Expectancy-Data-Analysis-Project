import os
import pickle
from flask import Flask, request, jsonify
import pandas as pd
import traceback

app = Flask(__name__)

# Load your trained model
with open("life_expectancy_model.pkl", "rb") as f:
    model = pickle.load(f)

# Home route
@app.route("/")
def home():
    return "<h2>Life Expectancy Prediction API is Running ✅</h2>"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        preds = model.predict(df)
        return jsonify({"predictions": preds.tolist()})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
