import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import pickle
import os

# Enable MLflow autologging (optional)
mlflow.sklearn.autolog()

# Load the dataset
data = pd.read_csv("data/iris.csv")

# Drop the 'Id' column and prepare features and target
data = data.drop(columns=["Id"])
X = data.drop("Species", axis=1)
y = data["Species"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start an MLflow experiment
mlflow.set_experiment("Iris Classification")

# Define the parameter combinations for 3 different runs
experiment_configs = [
    {"model_type": "RandomForest", "n_estimators": 50, "max_depth": 3},
    {"model_type": "RandomForest", "n_estimators": 100, "max_depth": 5},
    {"model_type": "LogisticRegression", "C": 1.0, "solver": "lbfgs"},
]

# Run experiments and track with MLflow
for config in experiment_configs:
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("n_estimators", config["n_estimators"])
        mlflow.log_param("max_depth", config["max_depth"])

        # Train the Random Forest model
        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Print results to console
        print(f"Config: {config}, Accuracy: {accuracy}")

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # Log the model
        mlflow.sklearn.log_model(
            model, artifact_path="random_forest_model"
        )

# Save the best model (optional: select based on inspection in MLflow UI)
best_model = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=42
)
best_model.fit(X_train, y_train)

# Ensure the output directory exists
os.makedirs("src", exist_ok=True)

with open("src/model.pkl", "wb") as f:
    pickle.dump(best_model, f)
