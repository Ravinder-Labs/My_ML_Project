import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# AWS ke liye application variable zaroori hai
application = Flask(__name__)
app = application

# Models load kar rahe hain (Read-Byte 'rb' mode mein)
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

## Route 1: Home Page
@app.route('/')
def index():
    return render_template('index.html')

## Route 2: Prediction Page (GET & POST)
@app.route('/predict_data', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Form se data extract karna
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # Data scaling (Sirf transform use karna hai, fit_transform nahi)
        new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        
        # Prediction
        result = ridge_model.predict(new_data_scaled)

        # Result ko home.html par bhejna
        rounded_result = round(result[0], 2)
        return render_template('home.html', result=rounded_result)
    
    else:
        # Agar GET request hai toh bas form dikhao
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)