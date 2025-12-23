import joblib
model = joblib.load("Monitoring dan Logging/model.pkl")
print(model.n_features_in_)

print(model.feature_names_in_)
