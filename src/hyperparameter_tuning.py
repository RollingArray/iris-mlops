"""
Author: Ranjoy Sen
Description: Hyperparameter optimization for Random Forest using Optuna.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import optuna
import pickle
import os
from optuna.visualization import plot_optimization_history

# Load the dataset
data = pd.read_csv("data/iris.csv")
data = data.drop(columns=["Id"])
X = data.drop("Species", axis=1)
y = data["Species"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Define objective function for Optuna
def objective(trial):
    """
    Objective function for Optuna to optimize hyperparameters of Random Forest.
    """
    # Define hyperparameter search space
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    # Train Random Forest with suggested hyperparameters
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Return evaluation metric
    return accuracy_score(y_test, y_pred)


# Create an Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Print and save the best hyperparameters
print(f"Best Hyperparameters: {study.best_params}")
print(f"Best Accuracy: {study.best_value}")

# Train and save the best model
best_model = RandomForestClassifier(**study.best_params, random_state=42)
best_model.fit(X_train, y_train)

# Ensure the output directory exists
os.makedirs("src", exist_ok=True)

# Save the best model
with open("src/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Best model saved as src/best_model.pkl")

# Use the study object from your optimization process
fig = plot_optimization_history(study)
fig.show()