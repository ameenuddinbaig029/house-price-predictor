from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model/house_price_model.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [
        float(request.form["longitude"]),
        float(request.form["latitude"]),
        float(request.form["housing_median_age"]),
        float(request.form["total_rooms"]),
        float(request.form["total_bedrooms"]),
        float(request.form["population"]),
        float(request.form["households"]),
        float(request.form["median_income"]),
        float(request.form["ocean_proximity"])
    ]
    prediction = model.predict([features])[0]
    return render_template("result.html", prediction=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)