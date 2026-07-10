import json
import os
import pickle
from datetime import datetime

import pandas as pd

from train_model import MODEL_PATH, train_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "logs", "activity_logs.json")


def load_model():
    if not os.path.exists(MODEL_PATH):
        train_model()

    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


def number_value(data, key, default=0):
    try:
        return float(data.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def prepare_features(data):
    current_hour = datetime.now().hour

    features = {
        "Temperature_C": number_value(data, "temperature"),
        "Humidity_%": number_value(data, "humidity"),
        "Light_Lux": number_value(data, "light"),
        "Appliance_Usage_kWh": number_value(data, "appliance_usage"),
        "HVAC_Usage_kWh": number_value(data, "hvac_usage"),
        "Water_Heater_kWh": number_value(data, "water_heater_usage"),
        "Total_Energy_kWh": number_value(data, "total_energy"),
        "Hour": number_value(data, "hour", current_hour),
        "Smoke_Detected": int(number_value(data, "smoke")),
        "Gas_Leak": int(number_value(data, "gas")),
    }

    return features


def apply_rule_engine(features, motion_prediction):
    triggered_rules = []

    actions = {
        "fan": "OFF",
        "light": "OFF",
        "alarm": "OFF",
        "mode": "Home",
    }

    priorities = {
        "Emergency": 1,
        "Security": 2,
        "Comfort": 3,
        "Energy Saving": 4,
    }

    highest_priority = priorities["Energy Saving"]

    if features["Smoke_Detected"] == 1 or features["Gas_Leak"] == 1:
        actions["alarm"] = "ON"
        actions["light"] = "ON"
        actions["fan"] = "OFF"
        actions["mode"] = "Emergency"
        highest_priority = priorities["Emergency"]

        if features["Smoke_Detected"] == 1:
            triggered_rules.append("Rule 4: Smoke detected, Emergency Mode")
        if features["Gas_Leak"] == 1:
            triggered_rules.append("Rule 5: Gas leak detected, Emergency Mode")
        triggered_rules.append("Rule 6: Emergency overrides all other rules")

        return actions, triggered_rules, highest_priority

    if motion_prediction == "Active" and features["Light_Lux"] < 300:
        actions["light"] = "ON"
        highest_priority = min(highest_priority, priorities["Comfort"])
        triggered_rules.append("Rule 1: Motion active and light below 300, Light ON")

    if motion_prediction == "Active" and features["Temperature_C"] > 28:
        actions["fan"] = "ON"
        highest_priority = min(highest_priority, priorities["Comfort"])
        triggered_rules.append("Rule 2: Motion active and temperature above 28, Fan ON")

    if motion_prediction == "Inactive":
        actions["fan"] = "OFF"
        actions["light"] = "OFF"
        actions["mode"] = "Energy Saving"
        highest_priority = min(highest_priority, priorities["Energy Saving"])
        triggered_rules.append("Rule 3: Motion inactive, Energy Saving Mode")

    if not triggered_rules:
        triggered_rules.append("No special rule triggered, normal Home Mode")

    return actions, triggered_rules, highest_priority


def save_log(sensor_inputs, motion_prediction, triggered_rules, actions):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as file:
            try:
                logs = json.load(file)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sensor_inputs": sensor_inputs,
        "ml_prediction": motion_prediction,
        "triggered_rules": triggered_rules,
        "final_device_actions": actions,
    }

    logs.append(log_entry)

    with open(LOG_PATH, "w", encoding="utf-8") as file:
        json.dump(logs, file, indent=4)

    return log_entry


def get_logs():
    if not os.path.exists(LOG_PATH):
        return []

    with open(LOG_PATH, "r", encoding="utf-8") as file:
        try:
            return json.load(file)
        except json.JSONDecodeError:
            return []


def make_prediction(data):
    model_data = load_model()
    features = prepare_features(data)

    feature_frame = pd.DataFrame([features], columns=model_data["feature_columns"])
    scaled_features = model_data["scaler"].transform(feature_frame)
    prediction_number = model_data["model"].predict(scaled_features)[0]
    motion_prediction = "Active" if prediction_number == 1 else "Inactive"

    actions, triggered_rules, priority_level = apply_rule_engine(features, motion_prediction)
    log_entry = save_log(features, motion_prediction, triggered_rules, actions)

    return {
        "motion_prediction": motion_prediction,
        "fan": actions["fan"],
        "light": actions["light"],
        "alarm": actions["alarm"],
        "mode": actions["mode"],
        "triggered_rules": triggered_rules,
        "priority_level": priority_level,
        "log_entry": log_entry,
    }
