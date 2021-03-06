# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 11:41:10 2020

@author: Adminr
"""
# importing the necessary dependencies
from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle
import os

app=Flask(__name__)# initializing a flask app
with open('CO2.pkl', 'rb') as handle:
    model = pickle.load(handle)
#filepath="C:\Users\LENOVO\internship\Flask\CO2.pkl"
#model=pickle.load(open(filepath,'rb'))

@app.route('/')# route to display the home page
def home():
    return render_template('home.html') #rendering the home page
@app.route('/Prediction',methods=['POST','GET'])
def prediction(): # route which will take you to the prediction page
    return render_template('indexnew.html')
@app.route('/Home',methods=['POST','GET'])
def my_home():
    return render_template('home.html')

@app.route('/predict',methods=["POST","GET"])# route to show the predictions in a web UI
def predict():
    #  reading the inputs given by the user
    input_feature=[float(x) for x in request.form.values() ]  
    features_values=[np.array(input_feature)]
    feature_name=['Make', 'Vehicle_Class', 'Engine_Size', 'Cylinders',
       'Transmission', 'Fuel_Type', 'Fuel_Consumption_City',
       'Fuel_Consumption_Hwy', 'Fuel_Consumption_Comb(mpg)']
    x=pd.DataFrame(features_values,columns=feature_name)
    
     # predictions using the loaded model file
    prediction=model.predict(x)  
    print("Prediction is:",prediction)
     # showing the prediction results in a UI
    return render_template("resultnew.html",prediction=prediction[0])
if __name__=="__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)
    #app.run(host='0.0.0.0', port=8000,debug=True)    # running the app
    