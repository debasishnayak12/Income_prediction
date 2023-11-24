import os
from flask import Flask, render_template, request
import joblib
import numpy as np

model = joblib.load(open("credit_card.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("front.html")

@app.route('/predict', methods=["POST"])
def inc_prediction():
   if request.method == "POST":
        data = request.form
        data1 = request.form["data1"]
        data2 = request.form["data2"]
        data3 = request.form["data3"]
        data = np.array([[data1, data2, data3]], dtype=float)

        # Add print statements to check the values
        print("data1:", data1)
        print("data2:", data2)
        print("data3:", data3)

      
        predict = float(model.predict(data)[0])  # Extracting the numerical value
        

        return render_template('result.html', predict_income="Customer's Income is {:.2f}".format(predict))


if __name__ == "__main__":
    app.run(debug=True)
