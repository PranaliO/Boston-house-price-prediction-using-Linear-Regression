from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("boston/boston_model.pkl")  # ✅ Make sure this file exists

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(request.form[key]) for key in request.form]
    prediction = model.predict([np.array(features)])
    return render_template("index.html", prediction_text=f"Predicted House Price: ${round(prediction[0]*1000, 2)}")

if __name__ == "__main__":
    app.run(debug=True)
