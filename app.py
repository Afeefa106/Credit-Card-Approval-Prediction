from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load model only
model = joblib.load("model/model.pkl")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        raw_data = request.form["inputData"]

        # Convert string → list of floats
        values = [float(x.strip()) for x in raw_data.split(",")]

        # Check for 30 features
        if len(values) != 30:
            return render_template("index.html", error="Enter exactly 30 values")

        data = np.array(values).reshape(1, -1)

        prediction = int(model.predict(data)[0])
        probability = model.predict_proba(data)[0]

        return render_template(
            "index.html",
            prediction=prediction,
            probability=probability
        )

    except Exception as e:
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True)