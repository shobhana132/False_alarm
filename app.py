import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from flask import Flask, jsonify, request


app = Flask(__name__)

@app.route('/train')
def train():
    df = pd.read_excel('False Alarm Cases.xlsx')
    df = df.iloc[:, 1:8]

    X = df.drop('Spuriosity Index(0/1)', axis=1)
    X.drop(['Unwanted substance deposition(0/1)', 'H2S Content(ppm)'], axis=1, inplace=True)
    y = df['Spuriosity Index(0/1)']

    ss = StandardScaler()
    scaled_array = ss.fit_transform(X)
    X = pd.DataFrame(scaled_array, columns=X.columns)

    model = KNeighborsClassifier()
    model.fit(X, y)
    joblib.dump(model, 'model.pkl')
    return jsonify({'message': 'model trained'})

@app.route('/test', methods = ['POST'])
def test():
    data = request.get_json()
    at = data['Temperature']
    cal = data['Calibration']
    hum = data['Humidity']
    nos = data['NoS']
    narr = np.array([at, cal, hum, nos]).reshape(1,4)
    X_test = pd.DataFrame(narr, columns=['Temperature', 'Callibration', 'Humiduty', 'No of sensors'])
    model = joblib.load('model.pkl')
    y_pred = model.predict(X_test)

    if y_pred == 0:
        return jsonify({'message': "no danger"})
    else:
        return jsonify({'message': "Dangerous"})

app.run(port = 5000)










