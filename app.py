import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import coe

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0])

    return render_template('index.html', prediction_text='Recommended selling price should be ${}'.format(output))


@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = round(prediction[0])
    return jsonify(output)


@app.route('/coe_title', methods=['GET'])
def get_coe_title():
    data = coe.coe_title_json
    return data


@app.route('/coe_prices', methods=['GET'])
def get_coe_prices():
    data = coe.coe_prices_json
    return data


if __name__ == "__main__":
    app.run(debug=True)
