from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model
with open("src/model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    # Expecting JSON payload with "features"
    data = request.json
    features = data["features"]

    # Predict species
    prediction = model.predict([features])
    return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
