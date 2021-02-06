import pickle
from flask import *
import numpy as np

model1 = pickle.load(open('regressor.pkl', 'rb'))

app = Flask(__name__)
@app.route('/')
def man():
    return render_template('rform.html')


@app.route('/predict', methods=['POST'])
def predict():
    val1 = request.form['length']
    val2 = request.form['diameter']
    val3 = request.form['height']
    val4 = request.form['sw']
    val5 = request.form['vw']
    val6 = request.form['shw']
    val7 = request.form['rings']
    val8 = request.form['sf']
    val9 = request.form['si']
    val10 = request.form['sm']
    arr = np.array([[val1,val2,val3,val4,val5,val6,val7,val8,val9,val10]])
    pred = model1.predict(arr)
    return render_template('result1.html', data1=pred)


if __name__ == "__main__":
    app.run()








# import pickle
# from flask import *
# import numpy as np

# model = pickle.load(open('titanicnb.pkl', 'rb'))

# app = Flask(__name__)
# @app.route('/')
# def man():
#     return render_template('Naive_Byes_form.html')


# @app.route('/predict', methods = ['POST'])
# def deploy():
#     val1 = request.form['gender']
#     val2 = request.form['age']
#     val3 = request.form['sib']
#     val4 = request.form['parch']
#     val5 = request.form['fare']
#     val6 = request.form['embc']
#     val7 = request.form['embq']
#     val8 = request.form['embs']
#     arr = np.array(val1,val2,val3,val4,val5,val6,val7,val8)
#     pred = model.predict(arr)
#     return render_template('result1.html', data1 = pred)


# if __name__ == "__main__":
#     app.run()