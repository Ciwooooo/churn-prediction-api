# package imports
import joblib
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os 
import json

# first writing in funciton-oriented to then potentially refactor into OOP

app = Flask(__name__)

# load the model and required artifacts at startup
MODEL_PATH = os.getenv('MODEL_PATH', 'models/churn_model.pkl') # fetches the os variable, if not present defaults to churn_model.pkl

model = joblib.load(MODEL_PATH)
label_encoders = joblib.load('models/label_encoders.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# load in model metadata
with open('models/model_metadata.json', 'r') as f:
    metadata = json.load(f)

# ----------------------------------------------------------
@app.route('/') # Flask-specific decorator, binds a func to an URL
def home():
    """Health check endpoint."""
    return jasonify({
        'status':'healthy',
        'service':'Churn Prediction API',
        'version':'1.0.0'
    })

# ----------------------------------------------------------
@app.route('/health')
def health():
    """Detailed health check"""
    return jsonify({
        'status':'healthy',
        'model_loaded': model is not None,
        'model_info': metadata
    })

# ----------------------------------------------------------
