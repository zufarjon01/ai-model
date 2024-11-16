from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(name)

# O'qitilgan modelni yuklash
model = joblib.load('lung_cancer_prediction_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
   # Modeldan ma'lumotlarni olish
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    smoking = int(request.form['smoking'])
    yellow_fingers = int(request.form['yellow_fingers'])
    anxiety = int(request.form['anxiety'])
    peer_pressure = int(request.form['peer_pressure'])
    chronic_disease = int(request.form['chronic_disease'])
    fatigue = int(request.form['fatigue'])
    allergy = int(request.form['allergy'])
    wheezing = int(request.form['wheezing'])
    alcohol_consuming = int(request.form['alcohol_consuming'])
    coughing = int(request.form['coughing'])
    shortness_of_breath = int(request.form['shortness_of_breath'])
    swallowing_difficulty = int(request.form['swallowing_difficulty'])
    chest_pain = int(request.form['chest_pain'])

    # Bashorat qilish uchun ma'lumotlarni tayyorlash
    features = np.array([[gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
                          chronic_disease, fatigue, allergy, wheezing, alcohol_consuming,
                          coughing, shortness_of_breath, swallowing_difficulty, chest_pain]])
    
    # Bashorat qilish
    prediction = model.predict(features)
    result = 'Lung Cancer Detected' if prediction[0] == 1 else 'No Lung Cancer Detected'

    return render_template('index.html', prediction=result)

if name== 'main':
    app.run(debug=True)