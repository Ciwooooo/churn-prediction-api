# package imports
from http.client import responses
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
    return jsonify({
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
@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict customer churn

    Expected JSON input:
    {
        "tenure": 12,
        "monthly_charges": 65.5,
        "total_charges": 786.0,
        "contract_type": "Month-to-month",
        "payment_method": "Electronic check",
        "internet_service": "Fiber optic",
        "tech_support": "No"
    }
    """
    # get json data
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error':'No data provided'}), 400
        
        # validate if required fields have been provided
        required_fields = feature_names
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            return jsonify({
                'error':'Missing required fields',
                'missing_fields': missing_fields
            }), 400
        
        # prepare input data
        input_df = pd.DataFrame([data])

        # do encoding for cat variables
        for col, encoder in label_encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = encoder.transform(input_df[col])
                except ValueError as e:
                    return jsonify({
                        'error':f'Invalid value for {col}',
                        'valid_values':encoder.classes._tolist()
                    }), 400
        
        # make sure the column order is correct 
        input_df = input_df[feature_names]

        # make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        # prepare response
        response = {
            'churn_prediction': int(prediction),
            'churn_probability': float(prediction_proba[1]),
            'risk_level':'High' if prediction_proba[1] > 0.7 else 'Medium' if prediction_proba[1] >0.4 else 'Low',
            'input_data': data
        }

        return jsonify(response)

        # error handling
    except Exception as e:
        return jsonify({'error':str(e)}), 500

# ----------------------------------------------------------
@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        data = request.get_json()

        if not data or 'customers' not in data:
            return jsonify({'error':'No customer data provided'}), 400
        
        customers = data['customers']

        # process each customer
        predictions = []
        for customer in customers:
            input_df = pd.DataFrame([customer])

            # encode cat variables
            for col, encoder in label_encoders.items():
                if col in input_df.columns:
                    input_df[col] = encoder.transform(input_df[col])
            
            # get the feature names and predict
            input_df = input_df[feature_names]

            pred = model.predict(input_df)[0]
            pred_proba = model.predict_proba(input_df)[0]

            predictions.append({
                'customer_daa':customer,
                'churn_prediction': int(pred),
                'churn_probability': float(pred_proba[1])
            })

        return jsonify({
            'predictions':predictions,
            'total_customers': len(predictions)
        })

    # error hanlding
    except Exceptions as e:
        return jsonify({'error':str(e)}), 500


# ----------------------------------------------------------
# start the flask app, listening on the specified port 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)