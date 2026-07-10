import json
import os
import pickle

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "smart_home_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "smart_home_model.pkl")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")

FEATURE_COLUMNS = [
    "Temperature_C",
    "Humidity_%",
    "Light_Lux",
    "Appliance_Usage_kWh",
    "HVAC_Usage_kWh",
    "Water_Heater_kWh",
    "Total_Energy_kWh",
    "Hour",
    "Smoke_Detected",
    "Gas_Leak",
]


def preprocess_data():
    data = pd.read_csv(DATA_PATH)

    for column in data.columns:
        if is_numeric_dtype(data[column]):
            data[column] = data[column].fillna(data[column].mean())
        else:
            data[column] = data[column].fillna(data[column].mode()[0])

    data["DateTime"] = pd.to_datetime(data["DateTime"])
    data["Hour"] = data["DateTime"].dt.hour
    data["Day_Night"] = data["Hour"].apply(lambda hour: "Day" if 6 <= hour < 18 else "Night")
    data["Motion_Sensor"] = data["Motion_Sensor"].map({"Inactive": 0, "Active": 1})
    data["Smoke_Detected"] = data["Smoke_Detected"].map({"No": 0, "Yes": 1})
    data["Gas_Leak"] = data["Gas_Leak"].map({"No": 0, "Yes": 1})

    return data


def create_visualizations(data, k_values, accuracy_values, y_test, y_pred):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, accuracy_values, marker="o")
    plt.title("KNN Accuracy for Different K Values")
    plt.xlabel("K Value")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, "accuracy_graph.png"), bbox_inches="tight")
    plt.close()

    matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Inactive", "Active"],
        yticklabels=["Inactive", "Active"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), bbox_inches="tight")
    plt.close()

    selected_features = ["Temperature_C", "Light_Lux", "Total_Energy_kWh", "Hour"]
    plt.figure(figsize=(12, 8))
    for index, feature in enumerate(selected_features, start=1):
        plt.subplot(2, 2, index)
        sns.histplot(data[feature], kde=True)
        plt.title(f"{feature} Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_distributions.png"), bbox_inches="tight")
    plt.close()

    results = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
    counts = results.value_counts().reset_index(name="Count")
    counts["Actual"] = counts["Actual"].map({0: "Inactive", 1: "Active"})
    counts["Predicted"] = counts["Predicted"].map({0: "Inactive", 1: "Active"})

    plt.figure(figsize=(7, 5))
    sns.barplot(data=counts, x="Actual", y="Count", hue="Predicted")
    plt.title("Motion Sensor Prediction Results")
    plt.savefig(os.path.join(PLOTS_DIR, "motion_prediction_results.png"), bbox_inches="tight")
    plt.close()

    usage_means = data[["Appliance_Usage_kWh", "HVAC_Usage_kWh", "Water_Heater_kWh"]].mean()
    plt.figure(figsize=(7, 5))
    sns.barplot(x=usage_means.index, y=usage_means.values)
    plt.title("Average Device Energy Usage")
    plt.ylabel("Average kWh")
    plt.xticks(rotation=20)
    plt.savefig(os.path.join(PLOTS_DIR, "device_usage.png"), bbox_inches="tight")
    plt.close()


def train_model():
    data = preprocess_data()

    x = data[FEATURE_COLUMNS]
    y = data["Motion_Sensor"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    k_values = [1, 3, 5, 7, 9]
    accuracy_values = []
    best_k = 1
    best_accuracy = 0
    best_model = None
    best_predictions = None

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train_scaled, y_train)
        predictions = model.predict(x_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        accuracy_values.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
            best_model = model
            best_predictions = predictions

    precision = precision_score(y_test, best_predictions, zero_division=0)
    recall = recall_score(y_test, best_predictions, zero_division=0)
    f1 = f1_score(y_test, best_predictions, zero_division=0)

    model_data = {
        "model": best_model,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "best_k": best_k,
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model_data, file)

    metrics = {
        "best_k": best_k,
        "accuracy": round(best_accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    create_visualizations(data, k_values, accuracy_values, y_test, best_predictions)

    print("Training completed successfully.")
    print(f"Best K: {best_k}")
    print(f"Accuracy: {best_accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    return metrics


if __name__ == "__main__":
    train_model()
