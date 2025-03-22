import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Import ridge and standard scaler pickle files
ridge_model = pickle.load(open('notebooks/models/ridge.pkl', 'rb'))
scaler_model = pickle.load(open('notebooks/models/scaler.pkl', 'rb'))


# Route for home page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/prediction', methods = ['GET', "POST"])
def prediction_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled = scaler_model.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html', result = result[0])
    
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
