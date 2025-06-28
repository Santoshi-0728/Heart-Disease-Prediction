from flask import Flask, render_template, request, session, send_file
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import joblib
import numpy as np
import io

app = Flask(__name__)
app.secret_key = 'your_secret_key'

model = joblib.load('model/heart_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/welcome', methods=['GET', 'POST'])
def welcome():
    if request.method == 'POST':
        session['name'] = request.form['name']
        session['age'] = request.form['age']
        session['gender'] = request.form['gender']
        return render_template('predict.html')
    return render_template('welcome.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        form = request.form

        input_array = np.array([
            int(form["male"]),
            int(form['age']),
            int(form['education']),
            int(form['currentSmoker']),
            int(form['cigsPerDay']),
            int(form['BPMeds']),
            int(form['prevalentStroke']),
            int(form['prevalentHyp']),
            int(form['diabetes']),
            float(form['totChol']),
            float(form['sysBP']),
            float(form['diaBP']),
            float(form['BMI']),
            float(form['heartRate']),
            float(form['glucose'])
        ])

        session['inputs'] = {
            'Gender (Male=1, Female=0)': form["male"],
            'Age': form['age'],
            'Education': form['education'],
            'Current Smoker': form['currentSmoker'],
            'Cigarettes Per Day': form['cigsPerDay'],
            'On BP Meds': form['BPMeds'],
            'Stroke History': form['prevalentStroke'],
            'Hypertension': form['prevalentHyp'],
            'Diabetes': form['diabetes'],
            'Total Cholesterol': form['totChol'],
            'Systolic BP': form['sysBP'],
            'Diastolic BP': form['diaBP'],
            'BMI': form['BMI'],
            'Heart Rate': form['heartRate'],
            'Glucose': form['glucose']
        }

        input_array = scaler.transform([input_array])
        prediction = model.predict(input_array)[0]

        result = 'Positive' if prediction == 1 else 'Negative'
        session['prediction'] = result

        return render_template("result.html", prediction=result,
                               name=session.get('name'),
                               age=session.get('age'),
                               gender=session.get('gender'))

    return render_template('predict.html')

@app.route('/download')
def download_pdf():
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    text = p.beginText(50, 750)
    text.setFont("Helvetica", 12)

    text.textLine("Heart Disease Prediction Report")
    text.textLine("-------------------------------")
    text.textLine(f"Name: {session.get('name')}")
    text.textLine(f"Age: {session.get('age')}")
    text.textLine(f"Gender: {session.get('gender')}")
    text.textLine(f"Prediction Result: {session.get('prediction')}")
    text.textLine("")
    text.textLine("Medical Inputs:")
    for key, value in session.get('inputs', {}).items():
        text.textLine(f"{key}: {value}")

    p.drawText(text)
    p.showPage()
    p.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name='prediction_result.pdf', mimetype='application/pdf')

if __name__ == '__main__':
    app.run(debug=True)
