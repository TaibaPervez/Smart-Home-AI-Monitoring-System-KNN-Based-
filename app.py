import json
import os

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from predict import get_logs, make_prediction
from train_model import METRICS_PATH, PLOTS_DIR, train_model


app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def health_check():
    return jsonify(
        {
            "message": "Smart Home AI Server is running",
            "status": "OK",
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({"error": "Please send JSON data"}), 400

        required_fields = [
            "temperature",
            "humidity",
            "light",
            "appliance_usage",
            "hvac_usage",
            "water_heater_usage",
            "total_energy",
            "smoke",
            "gas",
        ]

        missing_fields = []
        for field in required_fields:
            if field not in data:
                missing_fields.append(field)

        if missing_fields:
            return jsonify({"error": "Missing fields", "fields": missing_fields}), 400

        result = make_prediction(data)

        return jsonify(
            {
                "motion_prediction": result["motion_prediction"],
                "fan": result["fan"],
                "light": result["light"],
                "alarm": result["alarm"],
                "mode": result["mode"],
                "triggered_rules": result["triggered_rules"],
            }
        )

    except Exception as error:
        return jsonify({"error": str(error)}), 500


@app.route("/logs", methods=["POST", "GET"])
def logs():
    try:
        return jsonify(get_logs())
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@app.route("/train", methods=["POST"])
def train():
    try:
        metrics = train_model()
        return jsonify({"message": "Model trained successfully", "metrics": metrics})
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    try:
        if not os.path.exists(METRICS_PATH):
            train_model()

        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)

        return jsonify(data)
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@app.route("/plots/<filename>", methods=["GET"])
def plots(filename):
    return send_from_directory(PLOTS_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)
