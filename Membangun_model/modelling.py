import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
import dagshub
import os

# Connect MLflow to DagsHub
dagshub.init(repo_owner="Arensyifa", repo_name="mlsystem-spotify-aren", mlflow=True)

# Load dataset
train_df = pd.read_csv("Membangun_model/spotify_preprocessed_train.csv")
test_df = pd.read_csv("Membangun_model/spotify_preprocessed_test.csv")

X_train = train_df.drop("popular", axis=1)
y_train = train_df["popular"]

X_test = test_df.drop("popular", axis=1)
y_test = test_df["popular"]

with mlflow.start_run():

    # Model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Logging parameters & metrics manually
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("accuracy", acc)

    # Log the trained model
    mlflow.sklearn.log_model(model, "spotify_rf_model")

    # Confusion Matrix artifact
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

print("Model berhasil dilatih dan dilog ke MLflow DagsHub!")