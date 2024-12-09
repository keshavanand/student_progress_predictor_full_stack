# Preprocessing function for Persistence model
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

preprocessor = joblib.load('models/preprocessor_pipeline.pkl')

program_completion_model = tf.keras.models.load_model('models/gpa_prediction_model.keras')

def preprocess_data_completion(data):
    df = pd.DataFrame([data])

    # Extract the values and convert them to a NumPy 
    data = preprocessor.transform(df)

    #data = np.array(list(data.values()), dtype=float)
    prediction = program_completion_model.predict(data)

    
    return prediction[0][0]


print(preprocess_data_completion({'firstTermGpa': '3', 'secondTermGpa': '2', 'firstLanguage': '1', 'funding': '3', 
                                  'school': '2', 'fastTrack': '1', 'coop': '1', 'residency': '1', 'gender': '1', 
                                  'previousEducation': '1', 'ageGroup': '1', 'highSchoolAverageMark': '85', 'mathScore': '50', 'englishScore': '1'}))

