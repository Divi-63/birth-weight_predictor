from flask import Flask,request,jsonify, render_template
import pandas as pd 
import pickle
app = Flask(__name__)

def get_cleaned_data(form_data):
     gestation = float(form_data['gestation'])
     age = float(form_data['age'])
     parity = int(form_data['parity'])
     height = float(form_data['height'])
     weight = float(form_data['weight'])
     smoke = float(form_data['smoke'])

     cleaned_data = { "gestation":[gestation],
                     "parity":[parity],
                     "age": [age],
                     "height": [height],
                     "weight": [weight],
                     "smoke": [smoke]
                        }
     return cleaned_data
     

@app.route('/',methods=['GET'])
def home():
     return render_template("index.html")
#Define your Endpoint : 
@app.route("/predict",methods = ['POST'])
def get_prediction():
    #Get data from User :
    baby_data_form = request.form

    baby_data_cleaned = get_cleaned_data(baby_data_form)

    #Convert data into Dataframe : 
    baby_df= pd.DataFrame(baby_data_cleaned)
   

    # Load Machine Learning Trained Model:
    with open("model/model.pkl",'rb') as obj:
        model = pickle.load(obj)

    #Make Predictions : 
    prediction = model.predict(baby_df)[0]
    prediction = round(float(prediction),2)

    #return responsein JSON format :
    response = {"Prediction": prediction}

    return render_template("index.html",prediction=prediction) 

if __name__ == '__main__':
       app.run(debug=True)