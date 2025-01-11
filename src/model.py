import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

# Directory to save models locally
os.makedirs("models", exist_ok=True)

# Run experiments and track with MLflow
for i, config in enumerate(experiment_configs):
    with mlflow.start_run():
        # Log common parameters
        mlflow.log_param("model_type", config["model_type"])

        if config["model_type"] == "RandomForest":
            # Log Random Forest specific parameters
            mlflow.log_param("n_estimators", config["n_estimators"])
            mlflow.log_param("max_depth", config["max_depth"])

            # Train Random Forest model
            model = RandomForestClassifier(
                n_estimators=config["n_estimators"], max_depth=config["max_depth"], random_state=42
            )
        elif config["model_type"] == "LogisticRegression":
            # Log Logistic Regression specific parameters
            mlflow.log_param("C", config["C"])
            mlflow.log_param("solver", config["solver"])

            # Train Logistic Regression model
            model = LogisticRegression(C=config["C"], solver=config["solver"], random_state=42)

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Print results to console
        print(f"Run {i+1} - Config: {config}, Accuracy: {accuracy}")

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # Log the model to MLflow
        artifact_path = f"{config['model_type']}_model_{i+1}"
        mlflow.sklearn.log_model(model, artifact_path=artifact_path)

        # Save the model locally
        model_filename = f"models/{config['model_type'].lower()}_model_{i+1}.pkl"
        with open(model_filename, "wb") as f:
            pickle.dump(model, f)

# Save the best model (optional: replace with logic to select the best model)
best_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
best_model.fit(X_train, y_train)
with open("src/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
