import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize the Flask application
# Note: AWS Elastic Beanstalk expects the app instance to be named 'application'
application = Flask(__name__)
app = application

# Load the pre-trained Ridge regression model and Standard Scaler for inference
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Route: Application Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Route: Prediction Endpoint (Handles both form rendering and data processing)
@app.route('/predict_data', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Extract and parse input features from the submitted HTML form
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # Scale the incoming data using the pre-fitted Standard Scaler
        # Using 'transform' instead of 'fit_transform' to maintain consistency with training data
        new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        
        # Generate the Fire Weather Index (FWI) prediction
        result = ridge_model.predict(new_data_scaled)

        # Round the result to 2 decimal places and render it on the frontend
        rounded_result = round(result[0], 2)
        return render_template('home.html', result=rounded_result)
    
    else:
        # Handle GET request by rendering the default input form
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)