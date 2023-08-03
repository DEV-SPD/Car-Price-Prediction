import pickle
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder

with open('car_price_prediction', 'rb') as f:
    model = pickle.load(f)

def preprocessor(dataframe):
    categorical_features = ['name', 'company', 'fuel_type']
    for feature in categorical_features:
        encoder = LabelEncoder()
        dataframe[feature] = encoder.fit_transform(dataframe[feature])
    return dataframe

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_price():
    name = request.form.get('name')
    company = request.form.get('company')
    year = request.form.get('year')
    kms_driven = request.form.get('kms_driven')
    fuel_type = request.form.get('fuel_type')

    dict_ = {
        'name': [name],
        'company': [company],
        'year': [year],
        'kms_driven': [kms_driven],
        'fuel_type': [fuel_type]
    }
    df = pd.DataFrame(dict_, index=[0])
    print(df)
    df2 = preprocessor(df)

    result = model.predict(df2)
    return str(result)

if __name__ == '__main__':
    app.run(debug=True)
