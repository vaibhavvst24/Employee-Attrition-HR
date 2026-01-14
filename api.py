from fastapi import FastAPI
import pandas as pd
import joblib
import traceback

app = FastAPI()

model = joblib.load("employee_attrition_prediction.pkl")

@app.get("/")
def home():
    return {"message": "Employee Attrition Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])

        # Debug: show incoming columns
        print("Incoming data columns:", df.columns.tolist())

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return {
            "Attrition_Probability": f"{probability * 100:.2f}%",
            "Attrition_Prediction": "Yes" if prediction == 1 else "No"
        }

    except Exception as e:
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }
