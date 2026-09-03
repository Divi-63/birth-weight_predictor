import os
import pandas as pd
import pickle
from flask import Blueprint, request


path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model.pkl")
with open (path,'rb') as obj :
 model = pickle.load(obj)

predict_bp = Blueprint("predict",__name__)



def get_cleaned_data(form_data):
    gestation = float(form_data['gestation'])
    parity = int(form_data['parity'])
    age = float(form_data['age'])
    height = float(form_data['height'])
    weight = float(form_data['weight'])
    smoke = float(form_data['smoke'])

    cleaned_data = {"gestation":[gestation],
                    "parity":[parity],
                    "age":[age],
                    "height":[height],
                    "weight":[weight],
                    "smoke":[smoke]
                    }

    return cleaned_data







EXPECTED_COLUMNS = ["gestation","parity","age","height","weight","smoke"]





# define your endpoint
@predict_bp.route("/predict", methods = ['POST'])
def get_prediction():
    # get data from user
    # baby_data_form = request.form
    baby_data_form = request.get_json()


    # baby_data_cleaned = get_cleaned_data(baby_data_form)

    # convert into dataframe
    baby_df = pd.DataFrame(baby_data_form)
    baby_df = baby_df[EXPECTED_COLUMNS]

    # load machine leanring trained model
    # path = os.path.join(os.path.dirname(__file__), "model.pkl")
    # with open(path, 'rb') as obj:
    #     model = pickle.load(obj)
    # path = "C:\Users\divya\Documents\Flask\ML_Model\routes\model.pkl"
    # with open (path,'rb') as obj :
    #     model = pickle.load(obj)

    # make prediciton on user data
    prediction = model.predict(baby_df)
    prediction = round(float(prediction[0]), 2)

    # return reponse in a json format
    response = {"Prediction":prediction}

    # return render_template("index.html", prediction=prediction)
    return response