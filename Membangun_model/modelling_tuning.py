import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import dagshub
import json

# Connect MLflow
dagshub.init(repo_owner="Arensyifa", repo_name="mlsystem-spotify-aren", mlflow=True)

train_df = pd.read_csv("Membangun_model/spotify_preprocessed_train.csv")
test_df = pd.read_csv("Membangun_model/spotify_preprocessed_test.csv")

X_train = train_df.drop("popular", axis=1)
y_train = train_df["popular"]

X_test = test_df.drop("popular", axis=1)
y_test = test_df["popular"]

param_grid = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
}

search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=5,
    cv=3,
    scoring="accuracy",
    random_state=42
)

with mlflow.start_run():

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    preds = best_model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Manual logging
    mlflow.log_params(search.best_params_)
    mlflow.log_metric("best_accuracy", acc)

    # Save tuned model
    mlflow.sklearn.log_model(best_model, "spotify_rf_best_model")

    # Save best parameters
    with open("best_params.json", "w") as f:
        json.dump(search.best_params_, f)

    mlflow.log_artifact("best_params.json")

print("Tuning berhasil dan artefak sudah dikirim ke DagsHub!")