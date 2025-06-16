# Enhanced MLflow Experiment with Multiple Models and Parameters

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import pickle

# Load dataset
diabetes_dataset = pd.read_csv('data/diabetes.csv')

# Undersample to balance classes
df_class_0 = diabetes_dataset[diabetes_dataset["Outcome"] == 0]
df_class_1 = diabetes_dataset[diabetes_dataset["Outcome"] == 1]
df_class_0 = df_class_0.sample(len(df_class_1), random_state=42)
diabetes_dataset = pd.concat([df_class_0, df_class_1]).reset_index(drop=True)

# Features and target
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Train/test split
random_seed = 2
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=random_seed
)

# Define models and parameters
experiments = [
    {"name": "SVC_Linear", "model": SVC(kernel='linear', probability=True), "params": {"kernel": "linear"}},
    {"name": "SVC_RBF", "model": SVC(kernel='rbf', probability=True), "params": {"kernel": "rbf"}},
    {"name": "RandomForest_100", "model": RandomForestClassifier(n_estimators=100, random_state=random_seed), "params": {"n_estimators": 100}},
    {"name": "RandomForest_200", "model": RandomForestClassifier(n_estimators=200, random_state=random_seed), "params": {"n_estimators": 200}},
    {"name": "LogisticRegression", "model": LogisticRegression(max_iter=200), "params": {"solver": "lbfgs"}},
]
import os
import mlflow

# Force tracking inside the current working directory
mlflow_tracking_path = os.path.abspath("./mlruns")
print(f"Tracking to local path: {mlflow_tracking_path}")
mlflow.set_tracking_uri("file://" + mlflow_tracking_path)
os.makedirs(mlflow_tracking_path, exist_ok=True)

# Set MLflow experiment
mlflow.set_experiment("Diabetes_MultiModel_Experiment")

for exp in experiments:
    with mlflow.start_run(run_name=exp["name"]):
        model = exp["model"]
        model.fit(X_train, Y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics = {
            "train_accuracy": accuracy_score(Y_train, y_train_pred),
            "test_accuracy": accuracy_score(Y_test, y_test_pred),
            "test_precision": precision_score(Y_test, y_test_pred),
            "test_recall": recall_score(Y_test, y_test_pred),
            "test_f1": f1_score(Y_test, y_test_pred),
        }

        # Log parameters
        mlflow.log_params(exp["params"])
        mlflow.log_param("model_type", exp["name"])
        mlflow.log_param("random_seed", random_seed)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(
    model,
    artifact_path="model",
    input_example=X_test[:1]
)


        print(f"Logged {exp['name']} with Test Accuracy: {metrics['test_accuracy']:.4f}")

        os.makedirs("app", exist_ok=True)

# Save the best model (last model in the loop here)
with open("app/diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)