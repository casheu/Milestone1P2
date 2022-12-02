from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import joblib

# App Initialization
app = Flask(__name__)

# Load Pipeline Model
with open('pipeline.pkl', 'rb') as file_1:
  pipeline = joblib.load(file_1)

# Load Sequential Model
from tensorflow.keras.models import load_model
model_seq = load_model('model.h5')

@app.route("/")
def home():
    return "<h1>It Works!</h1>"

@app.route("/predict", methods=['POST'])
def titanic_predict():
    args = request.json

    data_inf = {
      'customerID': args.get('customerID'),
      'gender': args.get('gender'), 
      'SeniorCitizen': args.get('SeniorCitizen'), 
      'Partner': args.get('Partner'), 
      'Dependents': args.get('Dependents'),
      'PhoneService': args.get('PhoneService'), 
      'MultipleLines': args.get('MultipleLines'), 
      'InternetService': args.get('InternetService'), 
      'OnlineSecurity': args.get('OnlineSecurity'), 
      'OnlineBackup': args.get('OnlineBackup'),
      'DeviceProtection': args.get('DeviceProtection'),
      'TechSupport': args.get('TechSupport'), 
      'StreamingTV': args.get('StreamingTV'), 
      'StreamingMovies': args.get('StreamingMovies'), 
      'tenure': args.get('tenure'),
      'Contract': args.get('Contract'), 
      'PaperlessBilling': args.get('PaperlessBilling'), 
      'PaymentMethod': args.get('PaymentMethod'), 
      'MonthlyCharges': args.get('MonthlyCharges'), 
      'TotalCharges': args.get('TotalCharges')
    }
    print('[DEBUG] Data Inference : ', data_inf)
    
    # Transform Inference-Set
    data_inf = pd.DataFrame([data_inf])
    new_data_transform = pipeline.transform(data_inf)
    new_data_transform

    # Predict using Neural Network
    y_pred_inf = model.predict(new_data_transform)
    y_pred_inf = np.where(y_pred_inf >= 0.5, 1, 0)[0][0]
    if y_pred_inf == 0:
      label = 'Not Churned'
    else:
        label = 'Churned'
    
    print('[DEBUG] Result : ', y_pred_inf, label)
    print('')

    response = jsonify(
      result = str(y_pred_inf), 
      label_names = label)

    return response

if __name__ == "__main__":
    app.run(host='0.0.0.0')