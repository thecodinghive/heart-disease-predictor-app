# Dependencies used to build the web app
from flask import Flask, render_template, jsonify, request
from werkzeug.exceptions import HTTPException
import traceback
import io

# Used to load and run the model
# Packages: scikit-learn, joblib
from sklearn.externals import joblib
import numpy as np

# #import required packages
# import pandas as pd
#
# # will be used to split the data to train and test
# from sklearn.model_selection import train_test_split
# from sklearn import tree

# from sklearn import metrics  # will be used to calculate assessment metrics
# # will be used to compute confusion matrix and AUC
# from sklearn.metrics import classification_report
# # will be used to compute confusion matrix and AUC
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt  # will be used to generate ROC plot
# import seaborn as sns  # will be used to generate confusion matrix

app = Flask(__name__)
model = None


@app.route('/')
def form():
    return render_template('form.html')


@app.errorhandler(Exception)
def handle_error(e):
    code = 500
    if isinstance(e, HTTPException):
        code = e.code
    print("Error!", e)
    print(traceback.format_exc())
    return jsonify(error=str(e)), code


def load_model():
    global model
    # Only load model once
    if not model:
        print("------------------------>>>> loading model...")
        model = joblib.load("heart_classifier.pkl")
    return model


def valid_form_request():
    return request.method == "POST" and request.files.get("image")


"""
Calculates BMI given height (in kg), and weight (in cm)
BMI Formula: kg / m^2
Output is BMI, rounded to one decimal digit
"""


def calculate_bmi(height, weight):
    # Input height is in cm, so we divide by 100 to convert to metres
    return round(weight / ((height / 100) ** 2), 1)


"""
User our model to perform a prediction, given the input parameters.
Note that input to model is an array with the following parameters:
['ap_hi','ap_lo','age','cholesterol','bmi']

[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, bmi, bodyfat, lifestyle]
e.g.:
rf_model.predict([[48, 1, 165, 68, 110, 70, 0, 1, 0, 0, 1, 24.977043, 33.39809, 1]])
    => array([0])
    (or could be array([1]))
rf_model.predict_proba([[48, 1, 165, 68, 110, 70, 0, 1, 0, 0, 1, 24.977043, 33.39809, 1]])
    => array([[0.65566831, 0.34433169]])
"""


def predict(bp_high, bp_low, age, cholesterol, bmi):
    classifier = load_model()
    model_params = [[bp_high, bp_low, age, cholesterol, bmi]]
    result = {
        "prediction": classifier.predict(model_params)[0],
        "probabilities": classifier.predict_proba(model_params)[0]
    }
    return result


@app.route('/process_form', methods=["POST"])
def process_form():
    error = False
    if not valid_form_request():
        error = True

    age = int(request.form['age'])
    bpSystolic = int(request.form['bpSystolic'])
    bpDiastolic = int(request.form['bpDiastolic'])
    weight = float(request.form['weight'])  # kg
    height = float(request.form['height'])  # cm

    bmi = calculate_bmi(weight, height)

    # 0=Normal, 1=Above Normal, 2=Well Above Normal
    cholesterol = int(request.form['cholesterol'])
    cholesterolDescriptions = {
        0: "Normal",
        1: "Above Normal",
        2: "Well Above Normal",
    }

    # These are the values that we will display on the results page
    inputValues = {
        "Age": age,
        "Blood Pressure": "%s/%s" % (bpSystolic, bpDiastolic),
        "Weight": "%s kg" % weight,
        "Height": "%s cm" % height,
        "BMI": bmi,
        "Cholesterol": cholesterolDescriptions[cholesterol]
    }

    prediction = predict(bpSystolic, bpDiastolic, age, cholesterol, bmi)
    print(prediction)

    return render_template('results.html', prediction=prediction["prediction"], probabilities=prediction["probabilities"], inputValues=inputValues)


# if this is the main thread of execution, start the server
if __name__ == "__main__":
    print("* Starting Flask server..."
          "please wait until server has fully started")
    # debug=True options allows us to view our changes without restarting the server.
    app.run(host='0.0.0.0', debug=True)
