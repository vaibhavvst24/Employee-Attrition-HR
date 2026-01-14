import joblib
import pickle

model = joblib.load("employee_attrition_prediction.pkl")

print("Number of features:", model.n_features_in_)
print("Feature names used in training:\n")

print(model.feature_names_in_)

model = pickle.load(open("employee_attrition_prediction.pkl", "rb"))
print(type(model))

