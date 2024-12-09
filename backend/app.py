import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS
import joblib
import json

# Initialize the Flask application
app = Flask(__name__)

# Enable CORS for all routes and origins
CORS(app)

# Load the models
gpa_model = tf.keras.models.load_model('models/gpa_prediction_model.keras')
persistence_model = tf.keras.models.load_model('models/preistance_model.keras')
program_completion_model = tf.keras.models.load_model('models/completion_prediction_model.keras')

#Load preprocessor 
preprocessor = joblib.load('models/preprocessor_pipeline.pkl')


def reverse_min_max_normalization(normalized_value, min_value=0, max_value=4.5):
    # Reverse the min-max scaling
    return (normalized_value * (max_value - min_value)) + min_value


# Preprocessing function for Persistence model
def preprocess_data_persistence(data):
    pass

# Endpoint for predicting GPA
@app.route('/predict/gpa', methods=['POST'])
def predict_gpa():
    try:
        data = request.get_json()
       
        data = pd.DataFrame([data])

        data = preprocessor.transform(data)
        
        # Make a prediction using the GPA model
        prediction = gpa_model.predict(data[:, :4])

        print(prediction[0][0])
        
        prediction= reverse_min_max_normalization(prediction[0][0])
        print(prediction)
        # Return the result as a JSON response
        return jsonify({'gpa_prediction': float(prediction)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Endpoint for predicting persistence
@app.route('/predict/persistence', methods=['POST'])
def predict_persistence():
    try:
        data = request.get_json()

        data = pd.DataFrame([data])

        data = preprocessor.transform(data)

        prediction = (persistence_model.predict(data) > 0.5).astype(int)

        prediction = "Will be presisitance" if prediction[0][0] == 1 else "Not persistance"

        return jsonify({'persistence_prediction': prediction}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Endpoint for predicting program completion
@app.route('/predict/program-completion', methods=['POST'])
def predict_program_completion():
    try:
        data = request.get_json()
       
        data = pd.DataFrame([data])

        data = preprocessor.transform(data)
       
        prediction = (program_completion_model.predict(data) > 0.5).astype(int)

        prediction = "Completion" if prediction[0][0] == 1 else "Not completion"

        return jsonify({'program_completion_prediction': prediction}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
