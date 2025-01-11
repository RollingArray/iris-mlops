# Iris Classification Project

## **Overview**

This project demonstrates how to build, test, and deploy a machine learning model to classify iris flower species using the classic Iris dataset. The dataset contains measurements of iris flowers' sepal and petal dimensions, and the goal is to predict the species of an iris flower based on these measurements.

---

## **Project Structure**

```
iris-mlops/
├── .github/
│   └── workflows/
│       └── ci-cd.yml       # GitHub Actions CI/CD pipeline configuration
├── src/
│   ├── model.py            # Model training script
│   ├── app.py              # Flask API to serve the model
│   └── test_app.py         # Unit tests for the API
├── data/
│   └── iris.csv            # Dataset file
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .gitignore              # Ignored files and directories
```

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
3. **Deployment:** Placeholder for deploying the model (extendable for real-world deployment).

---

## **Technologies Used**

- **Python:** Programming language for model training and API development.
- **Flask:** Web framework for building the API.
- **scikit-learn:** Library for machine learning.
- **GitHub Actions:** CI/CD pipeline automation.

---

## **Future Enhancements**

- Deploy the API to a cloud platform (e.g., AWS, Azure, or Heroku).
- Add support for multiple model versions with version control.
- Implement additional evaluation metrics for model performance.

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

This project is licensed under the MIT License. See the `LICENSE` file for details.
