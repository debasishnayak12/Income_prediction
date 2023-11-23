import os
from flask import Flask,render_template,request
import pickle
import numpy as np
import joblib

model=joblib.load(open("credit_card.pkl","rb"))

app=Flask(__name__)
@app.route('/')
def home():
    return render_template("front.html")

@app.route('/predict',methods=["POST"])
def inc_prediction():
    data1=request.form("data1")
    data2=request.form("data2")
    data3=request.form("data3")
    data=np.array([[data1,data2,data3]])
    predict=model.predict(data)
    return render_template('result.html',predict_income="Coustomer's Income is {:.2f}".format(predict[0].item()))

if __name__=="__main__":
    app.run(debug=True)
    