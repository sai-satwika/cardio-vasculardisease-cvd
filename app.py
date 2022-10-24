from curses import echo
from optparse import Values
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

# ...
@app.route('/svm')
def svm():
    return render_template('svm.html')

# ...
@app.route('/ann/')
def ann():
    return render_template('ann.html')

@app.route('/knn')
def knn():
    return render_template('knn.html')

@app.route('/ensemble')
def ensemble():
    return render_template('ensemble.html')

# ...
@app.route("/predict", methods = ['POST', 'GET'])
def predict():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            # pred = predict(to_predict_list, to_predict_dict)
            values = to_predict_list
            print("form", to_predict_dict)
            print("values", values)
            model = pickle.load(open('heart.pkl', 'rb'))
            values = np.asarray(values)
            pred = model.predict(values.reshape(1, -1))[0]
            print("ans", pred)
            # return ans
            # print("pred", pred)
            
    except:
        message = "Please enter valid Data"
        return render_template("home.html", message = message)

    return render_template('predict.html', pred = pred)