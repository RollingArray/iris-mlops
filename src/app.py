from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model
with open("src/best_model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint to predict the iris species.
    Expects a JSON payload with a key "features" containing the input features.
    """
    # Expecting JSON payload with "features"
    data = request.json
    features = data["features"]

    # Predict species
    prediction = model.predict([features])
    return jsonify({"prediction": prediction[0]})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6000)
