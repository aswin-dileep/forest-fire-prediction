# app.py

from flask import Flask, render_template, request
from model.RandomForest_model import predict_forest_fire

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        wind_speed = float(request.form['wind_speed'])
        
        # Call the model to predict based on input
        prediction = predict_forest_fire(temperature, humidity, wind_speed)
        
        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
