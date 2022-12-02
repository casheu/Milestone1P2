from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import joblib

# App Initialization
app = Flask(__name__)

# Load Pipeline Model
with open('final_pipeline.pkl', 'rb') as file_1:
  model_pipeline = joblib.load(file_1)

# Load Sequential Model
from tensorflow.keras.models import load_model
model_seq = load_model('titanic_model.h5')

@app.route("/")
def home():
    return "<h1>It Works!</h1>"

@app.route("/predict", methods=['POST'])
def titanic_predict():
    args = request.json

    data_inf = {
      'PassengerId': args.get('PassengerId'),
      'Pclass': args.get('Pclass'), 
      'Name': args.get('Name'), 
      'Sex': args.get('Sex'), 
      'Age': args.get('Age'), 
      'SibSp': args.get('SibSp'),
      'Parch':  args.get('Parch'), 
      'Ticket': args.get('Ticket'), 
      'Fare': args.get('Fare'), 
      'Cabin': args.get('Cabin'), 
      'Embarked': args.get('Embarked')
    }
    print('[DEBUG] Data Inference : ', data_inf)
    
    # Transform Inference-Set
    data_inf = pd.DataFrame([data_inf])
    new_data_transform = model_pipeline.transform(data_inf)
    new_data_transform

    # Predict using Neural Network
    y_pred_inf = model_seq.predict(new_data_transform)
    y_pred_inf = np.where(y_pred_inf >= 0.5, 1, 0)[0][0]
    if y_pred_inf == 0:
      label = 'Not Survived'
    else:
        label = 'Survived'
    
    print('[DEBUG] Result : ', y_pred_inf, label)
    print('')

    response = jsonify(
      result = str(y_pred_inf), 
      label_names = label)

    return response

if __name__ == "__main__":
    app.run(host='0.0.0.0')