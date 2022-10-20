from curses import echo
from optparse import Values
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
# read our pickle file and label our logisticmodel as model
model = pickle.load(open('heart.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('svm.html')

# @app.route('/predict',methods=['POST','GET'])

def predict(values, dic):

    model = pickle.load(open('heart.pkl', 'rb'))
    values = np.asarray(values)
    print("values", values)
    return model.predict(values.reshape(1, -1))[0]

    # int_features = [float(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)
    # print(prediction)
    # # echo(prediction)
    # if prediction==0:
    #     return render_template('svm.html',
    #                            prediction_text='Low chances of patient having diabetes'.format(prediction),
    #                            )
    # else:
    #     return render_template('svm.html',
    #                            prediction_text='High chances of patient having diabetes'.format(prediction),
    #                           )

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
            ans = model.predict(values.reshape(1, -1))[0]
            print("ans", ans)
            # return ans
            # print("pred", pred)
    except:
        message = "Please enter valid Data"
        return render_template("home.html", message = message)

    return render_template('index1.html')

if __name__ == "__main__":
    app.run(debug=True)