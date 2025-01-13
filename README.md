# Iris Classification MLOps Project

## **Overview**

This project demonstrates how to build, test, and deploy a machine learning model to classify iris flower species using the classic Iris dataset. The dataset contains measurements of iris flowers' sepal and petal dimensions, and the goal is to predict the species of an iris flower based on these measurements.

---

## **Folder and File Structure**

```
Iris-mlops
├── requirements.txt
├── src/
│   ├── app.py
│   ├── best_model.pkl
│   ├── dynamic_add_data_to_iris.py
│   ├── hyperparameter_tuning.py
│   ├── model.pkl
│   ├── model.py
│   ├── response.json
│   ├── test_app.py
├── DockerFile
├── requirements.txt
├── README.md
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci-cd.yml
```

---

### **1. requirements.txt**
A list of Python dependencies required to run the project. Includes libraries for data processing, machine learning, and deployment.

---

### **2. src/**
Contains the core scripts and resources for the project.

- **app.py**: Flask or FastAPI application script for serving the trained model via REST API.
- **best_model.pkl**: The best-performing trained model from hyperparameter tuning, ready for deployment.
- **dynamic_add_data_to_iris.py**: Script for dynamically adding new data to the Iris dataset.
- **hyperparameter_tuning.py**: Script for optimizing model hyperparameters using Optuna.
- **model.pkl**: Generated trained model for testing and evaluation.
- **model.py**: Contains the core logic for model training and evaluation.
- **response.json**: Sample API response for testing the app endpoints.
- **test_app.py**: Unit tests for validating the API functionality and model performance.

---

### **3. DockerFile**
Defines the Docker image for the project, specifying the environment setup, dependencies, and how to run the application in a containerized environment.

---

### **4. README.md**
Comprehensive guide to the project, icluding installation steps, usage instructions, and details about the workflow.

---

### **5. .gitignore**
Specifies files and directories to be ignored by Git, such as logs, `.pkl` files, and virtual environment folders.

---

### **6. .github/workflows/**
Contains CI/CD pipeline configuration files.

- **ci-cd.yml**: GitHub Actions workflow to automate testing, building, and deployment processes for the project.

---

## **Dataset Description**

The dataset used in this project is the Iris dataset, which contains measurements of three iris flower species:

- **Iris-setosa**
- **Iris-versicolor**
- **Iris-virginica**

**Features:**

- **SepalLengthCm**: Sepal length in centimeters.
- **SepalWidthCm**: Sepal width in centimeters.
- **PetalLengthCm**: Petal length in centimeters.
- **PetalWidthCm**: Petal width in centimeters.

**Target Variable:**

- **Species**: The species of the iris flower.

**Sample Data:**

```
Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species
1,5.1,3.5,1.4,0.2,Iris-setosa
2,4.9,3.0,1.4,0.2,Iris-setosa
3,4.7,3.2,1.3,0.2,Iris-setosa
4,4.6,3.1,1.5,0.2,Iris-setosa
```

---

## **Project Workflow**

### **1. Data Preparation**

- The dataset is loaded from `data/iris.csv`.
- The `Id` column is removed, and the remaining columns are split into features (`SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`) and the target variable (`Species`).

### **2. Model Training**

- A **Random Forest Classifier** is trained to predict the flower species.
- The model is evaluated using accuracy and saved as `model.pkl` for deployment.

### **3. API Development**

- A **Flask API** is created to serve the trained model. It accepts JSON input containing flower measurements and returns the predicted species.

### **4. Testing**

- Unit tests are written using Python's `unittest` framework to ensure the API functions as expected.

### **5. CI/CD Pipeline**

- A **GitHub Actions** pipeline automates the following stages:
  - **Linting:** Ensures code quality using `flake8`.
  - **Testing:** Runs unit tests to verify functionality.
  - **Deployment:** Simulates deployment (can be extended to deploy to cloud platforms).

---

## **How to Run the Project**

### **1. Prerequisites**

Ensure you have the following installed:

- Python 3.8+
- pip (Python package installer)

### **2. Clone the Repository**

```bash
git clone <repository-link>
cd iris-mlops
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Train the Model**

```bash
python src/model.py
```

This will generate `model.pkl` in the `src/` directory.

### **5. Run the API**

```bash
python src/app.py
```

The API will start at `http://127.0.0.1:5000/`.

### **6. Test the API**

Use a tool like Postman or `curl` to test the `/predict` endpoint.

Example `curl` command:

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

Expected Output:

```json
{
    "prediction": "Iris-setosa"
}
```

---

## **Continuous Integration/Continuous Deployment (CI/CD)**

A GitHub Actions pipeline is configured to automate code quality checks, testing, and deployment. The configuration is in `.github/workflows/ci-cd.yml`.

### Pipeline Stages:

1. **Linting:** Ensures code adheres to Python standards using `flake8`.
2. **Testing:** Runs the unit tests from `test_app.py`.
3. **Deployment:** Placeholder for deploying the model.

---

## **Technologies Used**

- **Python:** Programming language for model training and API development.
- **Flask:** Web framework for building the API.
- **scikit-learn:** Library for machine learning.
- **GitHub Actions:** CI/CD pipeline automation.

---

## **Contributing**

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

---

## **License**

This project is licensed under the MIT License.