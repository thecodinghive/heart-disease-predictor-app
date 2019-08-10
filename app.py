# Dependencies used to build the web app
from flask import Flask, render_template, jsonify, request
from werkzeug.exceptions import HTTPException
import traceback
import io

# Used to load and run the model
# Packages: scikit-learn, joblib
from sklearn.externals import joblib
import numpy as np

app = Flask(__name__)
model = None

# Homepage: The heart health form
@app.route('/')
def form():
    # Get the values if specified in the URL (so we can edit them)
    values = request.values

    return render_template('form.html', form_values=values)


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
        print("--->>>> loading model...")
        model = joblib.load("heart_classifier.pkl")
    return model


def valid_form_request():
    return request.method == "POST" and request.files.get("image")


def calculate_bmi(height_cm, weight_kg):
    """
    Calculates BMI given height (in kg), and weight (in cm)
    BMI Formula: kg / m^2
    Output is BMI, rounded to one decimal digit
    """

    # Input height is in cm, so we divide by 100 to convert to metres
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 1)


def predict(bp_high, bp_low, age, cholesterol, bmi):
    """
    User our model to perform a prediction, given the input parameters.
    Note that input to model is a 2d array with the following parameters:
        [['ap_hi','ap_lo','age','cholesterol','bmi']]
    """

    classifier = load_model()
    # Note that our classifier expects a 2D array
    model_params = [[bp_high, bp_low, age, cholesterol, bmi]]
    result = {
        # classifier.predict returns an array containing the prediction
        #   e.g. => [[0]]
        "prediction": classifier.predict(model_params)[0],
        # classifier.predict_proba returns an array containing the probabilities of each class
        #   e.g. => [[0.65566831, 0.34433169]]
        "probabilities": classifier.predict_proba(model_params)[0]
    }
    return result


def get_form_values():
    # https://stackoverflow.com/a/16664376/76710
    form_values = {
        'age': int(request.form['age']),
        'bp_systolic': int(request.form['bp_systolic']),
        'bp_diastolic': int(request.form['bp_diastolic']),
        'weight_kg': float(request.form['weight_kg']),
        'height_cm': float(request.form['height_cm']),
        'cholesterol': int(request.form['cholesterol'])
    }
    return form_values


@app.route('/process_form', methods=["POST"])
def process_form():
    # Error handling
    # error = False
    # if not valid_form_request():
    #     error = True

    values = get_form_values()
    bmi = calculate_bmi(values['height_cm'], values['weight_kg'])

    # 0=Normal, 1=Above Normal, 2=Well Above Normal
    cholesterol_descriptions = {
        0: "Normal",
        1: "Above Normal",
        2: "Well Above Normal",
    }

    # These are the values that we will display on the results page
    input_values = {
        "Age": values['age'],
        "Blood Pressure": "%s/%s" % (values['bp_systolic'], values['bp_diastolic']),
        "Weight": "%s kg" % values['weight_kg'],
        "Height": "%s cm" % values['height_cm'],
        "BMI": bmi,
        "Cholesterol": cholesterol_descriptions[values['cholesterol']]
    }

    prediction = predict(
        values['bp_systolic'],
        values['bp_diastolic'],
        values['age'],
        values['cholesterol'],
        bmi
    )
    return render_template('results.html', prediction=prediction["prediction"], probabilities=prediction["probabilities"], input_values=input_values, form_values=values)


# Start the server
if __name__ == "__main__":
    print("* Starting Flask server..."
          "please wait until server has fully started")
    # debug=True options allows us to view our changes without restarting the server.
    app.run(host='0.0.0.0', debug=True)
