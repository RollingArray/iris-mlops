import pandas as pd
import random

# Load the existing dataset
dataset_path = "data/iris.csv"
iris_data = pd.read_csv(dataset_path)

# Dynamically generate new rows
num_rows_to_add = 5  # Number of new rows to add per run
new_data = []

# Generate unique IDs based on the current dataset
start_id = iris_data["Id"].max() + 1

species_choices = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
for i in range(num_rows_to_add):
    new_row = {
        "Id": start_id + i,
        "SepalLengthCm": round(random.uniform(4.5, 8.0), 1),
        "SepalWidthCm": round(random.uniform(2.0, 4.5), 1),
        "PetalLengthCm": round(random.uniform(1.0, 6.9), 1),
        "PetalWidthCm": round(random.uniform(0.1, 2.5), 1),
        "Species": random.choice(species_choices),
    }
    new_data.append(new_row)

# Convert the list of new rows to a DataFrame
new_data_df = pd.DataFrame(new_data)

# Append the new data to the existing dataset
updated_iris_data = pd.concat([iris_data, new_data_df], ignore_index=True)

# Save the updated dataset back to the same file
updated_iris_data.to_csv(dataset_path, index=False)

print(
    f"{num_rows_to_add} new rows added successfully. "
    f"Updated dataset saved to {dataset_path}."
)
