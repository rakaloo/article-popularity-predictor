import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.externals import joblib
from werkzeug import exceptions

from pipe import DayOfWeekTransformer, MonthTransformer, select_time_column, select_text_column


app = Flask(__name__)


@app.route('/model', methods=['POST'])
def get_prediction():
    try:
        timestamp = request.form['timestamp']
        description = request.form['description']
    except exceptions.BadRequest:
        raise exceptions.BadRequest('Post request failed due to missing timestamp or description')

    post_data = np.array([[pd.Timestamp(timestamp), description]])
    pipe = joblib.load('pipe.pkl')
    prediction = pipe.predict(post_data)[0]

    return jsonify({'prediction': prediction})
