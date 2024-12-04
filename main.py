import pandas as pd
import numpy as np
import templates as templates
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, request, render_template
import pickle
df = pd.read_csv('flats_moscow.csv')
dset = df.values
Y = dset[:, 1]
X = dset[:, 2:]
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)
regr = RandomForestRegressor()
regr.fit(x_train, y_train)
pickle.dump(regr, open('model.sav', 'wb'))
app = Flask('__name__')
q = ""


@app.route("/")
@app.route("/apartprice")
def loadPage():
    return render_template('home.html', query="")


@app.route("/predict", methods=['POST'])
def predict():
    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    inputQuery7 = request.form['query7']
    inputQuery8 = request.form['query8']
    inputQuery9 = request.form['query9']
    ml_model = pickle.load(open('model.sav', 'rb'))
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4,
             inputQuery5, inputQuery6, inputQuery7, inputQuery8, inputQuery9]]
    pred = ml_model.predict(data)
    return render_template('home.html', output1=pred[0],
                           query1=request.form['query1'], query2=request.form['query2'],
                           query3=request.form['query3'], query4=request.form['query4'],
                           query5=request.form['query5'], query6=request.form['query6'],
                           query7=request.form['query7'], query8=request.form['query8'],
                           query9=request.form['query9'])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)

