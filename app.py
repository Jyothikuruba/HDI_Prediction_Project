from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("models/hdi_model.pkl")

def classify_hdi(score):
    if score >= 0.800:
        return "Very High"
    elif score >= 0.700:
        return "High"
    elif score >= 0.550:
        return "Medium"
    else:
        return "Low"

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    category = None

    if request.method == "POST":
        life = float(request.form["life"])
        mean = float(request.form["mean"])
        expected = float(request.form["expected"])
        gni = float(request.form["gni"])

        features = np.array([[life, mean, expected, gni]])

        pred = model.predict(features)[0]

        if isinstance(pred, str):
            prediction = pred
            category = pred
        else:
            prediction = round(float(pred), 4)
            category = classify_hdi(prediction)

    return render_template(
        "index.html",
        prediction=prediction,
        category=category
    )

if __name__ == "__main__":
    app.run(debug=True)