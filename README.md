# 🧠 Employee Attrition Prediction for HR

A Machine Learning powered HR Analytics system that predicts employee attrition risk and helps HR teams proactively identify and retain at-risk employees using data-driven insights.

## 📌 Project Overview

Employee attrition leads to high hiring costs, loss of productivity, and disruption in business operations.
This project uses Machine Learning to predict which employees are likely to leave the organization and provides probability-based risk scoring that HR teams can act on.

The system is deployed using a Flask Web Application, allowing HR users to enter employee details and receive real-time attrition risk predictions.

## 🚀 Key Objectives

Predict whether an employee is likely to leave

Provide probability-based risk scores

Identify key drivers of attrition

Make the model usable by HR teams through a web interface

## 🗂 Dataset

The dataset is based on a real-world HR employee dataset with features such as:

Demographics (Age, Gender, Marital Status)

Job details (Department, Role, Job Level, Work Life Balance)

Performance and satisfaction metrics

Salary, promotions, and working history

## 🧪 Machine Learning Pipeline

Data Cleaning & Encoding

Feature Selection

Class Imbalance Handling

Random Forest Model Training

Threshold Tuning to improve Attrition Recall

Model Evaluation using Confusion Matrix, Precision, Recall, F1-Score, AUC

## 🖥 Web Application (Flask, FastAPI)

A user-friendly web interface allows HR users to:

Enter employee details

Get Attrition Risk (High / Low)

View Attrition Probability (%)

The application ensures:

Correct feature mapping

Proper label encoding

Model-consistent feature ordering

## 🧩 Input Features Used

Age
BusinessTravel
Department
DistanceFromHome
Education
EducationField
EmployeeCount
EnvironmentSatisfaction
Gender
JobInvolvement
JobRole
JobSatisfaction
MaritalStatus
MonthlyIncome
NumCompaniesWorked
OverTime
PercentSalaryHike
RelationshipSatisfaction
StockOptionLevel
TotalWorkingYears
TrainingTimesLastYear
WorkLifeBalance

## 🛠 Tech Stack

Python

Pandas, NumPy

Scikit-learn

Flask

HTML, CSS

Pickle (Model Serialization)

## 📈 Business Value

This system allows HR teams to:

Identify high-risk employees early

Take preventive actions (salary review, promotion, workload balancing)

Reduce employee churn

Improve workforce stability

# Flask Webpage 

![Employee Attrition Form](screenshots/eap1.png)
![Attrition Prediction Output](screenshots/eap2.png)
