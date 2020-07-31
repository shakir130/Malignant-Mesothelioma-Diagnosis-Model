import numpy as np
from flask import Flask, request, jsonify, render_template,url_for
import pickle

app = Flask(__name__)
model = pickle.load(open('finalized_model.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    # prediction = model.predict([[-1.190760, 1.430058, 0.682688]])
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if output > 0:
        prediction_text ='Healthy'
    else:
        prediction_text = 'Patient'

    return render_template('index.html', prediction_text='The Record you entered is Patient {}'.format(prediction_text))


@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)