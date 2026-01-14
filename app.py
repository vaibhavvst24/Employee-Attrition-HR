from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("employee_attrition_prediction.pkl", "rb"))

# Label encoders used during training
gender_map = {"Male": 1, "Female": 0}
marital_map = {"Single": 2, "Married": 1, "Divorced": 0}
department_map = {
    "Sales": 0,
    "Research & Development": 1,
    "Human Resources": 2
}
jobrole_map = {
    "Sales Executive": 0,
    "Research Scientist": 1,
    "Laboratory Technician": 2,
    "Manufacturing Director": 3,
    "Healthcare Representative": 4
}
travel_map = {
    "Travel_Rarely": 2,
    "Travel_Frequently": 1,
    "Non-Travel": 0
}
overtime_map = {"Yes": 1, "No": 0}
education_field_map = {
    "Life Sciences": 0,
    "Medical": 1,
    "Marketing": 2,
    "Technical Degree": 3,
    "Human Resources": 4
}

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect inputs
        age = int(request.form["Age"])
        travel = travel_map[request.form["BusinessTravel"]]
        department = department_map[request.form["Department"]]
        distance = int(request.form["DistanceFromHome"])
        education = int(request.form["Education"])
        edufield = education_field_map[request.form["EducationField"]]
        envsat = int(request.form["EnvironmentSatisfaction"])
        gender = gender_map[request.form["Gender"]]
        jobinv = int(request.form["JobInvolvement"])
        jobrole = jobrole_map[request.form["JobRole"]]
        jobsat = int(request.form["JobSatisfaction"])
        marital = marital_map[request.form["MaritalStatus"]]
        income = int(request.form["MonthlyIncome"])
        companies = int(request.form["NumCompaniesWorked"])
        overtime = overtime_map[request.form["OverTime"]]
        hike = int(request.form["PercentSalaryHike"])
        relsat = int(request.form["RelationshipSatisfaction"])
        stock = int(request.form["StockOptionLevel"])
        totalyrs = int(request.form["TotalWorkingYears"])
        training = int(request.form["TrainingTimesLastYear"])
        worklife = int(request.form["WorkLifeBalance"])

        # IBM dataset constant
        employee_count = 1

        # Feature array in training order
        features = np.array([[ 
            age,
            travel,
            department,
            distance,
            education,
            edufield,
            employee_count,
            envsat,
            gender,
            jobinv,
            jobrole,
            jobsat,
            marital,
            income,
            companies,
            overtime,
            hike,
            relsat,
            stock,
            totalyrs,
            training,
            worklife
        ]])

        # Model prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] * 100

        result = "High Risk" if prediction == 1 else "Low Risk"

        return render_template(
            "index.html",
            prediction=result,
            probability=f"{probability:.2f}%"
        )

    except Exception as e:
        return render_template("index.html", prediction="Error", probability=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

