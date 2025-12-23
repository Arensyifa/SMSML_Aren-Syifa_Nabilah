import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
import dagshub

# Connect MLflow to DagsHub
dagshub.init(
    repo_owner="Arensyifa",
    repo_name="mlsystem-spotify-aren",
    mlflow=True
)

mlflow.sklearn.autolog()

# Load dataset
train_df = pd.read_csv("Membangun_model/dataset_preprocessing/spotify_preprocessed_train.csv")
test_df = pd.read_csv("Membangun_model/dataset_preprocessing/spotify_preprocessed_test.csv")

X_train = train_df.drop("popular", axis=1)
y_train = train_df["popular"]

X_test = test_df.drop("popular", axis=1)
y_test = test_df["popular"]

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")

print("Training selesai & logged via MLflow Autologging ✅")