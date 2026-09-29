from flask import Flask, render_template,request
import joblib
import numpy as np


app=Flask(__name__)

# load train model
model=joblib.load("iris_model.pkl")
target_names=["Setosa","Versicolor","Virginica"]

@app.route("/")
def home():
    return render_template("index.html",title="Iris Classifier")

@app.route("/predict",methods=["POST"])

def predict():
    # get value from FORM
    sepal_length=float(request.form["sepal_length"])
    sepal_width=float(request.form["sepal_width"])
    petal_length=float(request.form["petal_length"])
    petal_width=float(request.form["petal_width"])


    # make predition
    features = np.array([[sepal_length, sepal_width,petal_length,petal_width]])


    prediction=model.predict(features)[0]
    result=target_names[prediction]

    return render_template("result.html",result=result,title="prediction")

if __name__ == "__main__":
  app.run(debug=True)
