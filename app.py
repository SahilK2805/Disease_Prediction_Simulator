from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Initialize Disease Prediction Variables
l1 = [
    'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
    'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
    'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
    'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
    'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
    'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
    'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
    'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
    'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
    'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
    'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
    'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
    'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria',
    'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
    'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding',
    'distention_of_abdomen', 'history_of_alcohol_consumption', 'blood_in_sputum',
    'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads',
    'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
    'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
]

disease = [
    'Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
    'Peptic ulcer diseae', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension',
    'Migraine', 'Cervical spondylosis',
    'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
    'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis',
    'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)',
    'Heartattack', 'Varicoseveins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis',
    'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis',
    'Impetigo'
]

# Initialize symptoms list with zeros
l2 = [0] * len(l1)

# Load and preprocess Training Data
df = pd.read_csv("Training.csv")

df.replace({
    'prognosis': {
        'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
        'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8, 'Bronchial Asthma': 9,
        'Hypertension ': 10, 'Migraine': 11, 'Cervical spondylosis': 12,
        'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16,
        'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19, 'Hepatitis B': 20, 'Hepatitis C': 21,
        'Hepatitis D': 22, 'Hepatitis E': 23, 'Alcoholic hepatitis': 24, 'Tuberculosis': 25,
        'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29,
        'Varicose veins': 30, 'Hypothyroidism': 31, 'Hyperthyroidism': 32, 'Hypoglycemia': 33,
        'Osteoarthristis': 34, 'Arthritis': 35,
        '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37, 'Urinary tract infection': 38,
        'Psoriasis': 39, 'Impetigo': 40
    }
}, inplace=True)

X = df[l1]
y = df[["prognosis"]]
np.ravel(y)

# Load and preprocess Testing Data
tr = pd.read_csv("Testing.csv")
tr.replace({
    'prognosis': {
        'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
        'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8, 'Bronchial Asthma': 9,
        'Hypertension ': 10, 'Migraine': 11, 'Cervical spondylosis': 12,
        'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16,
        'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19, 'Hepatitis B': 20, 'Hepatitis C': 21,
        'Hepatitis D': 22, 'Hepatitis E': 23, 'Alcoholic hepatitis': 24, 'Tuberculosis': 25,
        'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29,
        'Varicose veins': 30, 'Hypothyroidism': 31, 'Hyperthyroidism': 32, 'Hypoglycemia': 33,
        'Osteoarthristis': 34, 'Arthritis': 35,
        '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37, 'Urinary tract infection': 38,
        'Psoriasis': 39, 'Impetigo': 40
    }
}, inplace=True)

X_test = tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)

# Initialize classifiers
clf_tree = tree.DecisionTreeClassifier().fit(X, y)
clf_rf = RandomForestClassifier().fit(X, y)
clf_nb = GaussianNB().fit(X, y)

# Function to predict disease
def predict_disease(algorithm, symptoms):
    input_symptoms = [1 if symptom in symptoms else 0 for symptom in l1]
    input_test = [input_symptoms]

    if algorithm == 'DecisionTree':
        prediction = clf_tree.predict(input_test)[0]
    elif algorithm == 'RandomForest':
        prediction = clf_rf.predict(input_test)[0]
    elif algorithm == 'NaiveBayes':
        prediction = clf_nb.predict(input_test)[0]
    else:
        return "Invalid Algorithm"

    return disease[prediction]

# Configure the Gemini API client
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    raise ValueError("API key not found! Please set the GOOGLE_API_KEY in your environment.")

# Initialize the Gemini Pro model and chat session
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

# Function to get a response from the Gemini model
def get_gemini_response(question):
    try:
        response = chat.send_message(question, stream=True)
        return ''.join([chunk.text for chunk in response])
    except Exception as e:
        return f"Error: {str(e)}"

# Route for the main page (Disease Prediction)
@app.route('/')
def index():
    return render_template('index.html', symptoms=l1)

# Route to handle disease prediction
@app.route('/predict', methods=['POST'])
def predict():
    selected_symptoms = request.form.getlist('symptoms')
    algorithm = request.form['algorithm']
    predicted_disease = predict_disease(algorithm, selected_symptoms)
    return jsonify({"predicted_disease": predicted_disease})

# New Route: /sanky for Gemini AI Chat
@app.route('/sanky', methods=['GET', 'POST'])
def sanky():
    if request.method == 'POST':
        data = request.get_json()
        question = data.get('question', '')
        if question:
            response = get_gemini_response(question)
            return jsonify({"response": response})
        else:
            return jsonify({"error": "Please provide a question."}), 400
    return render_template('sanky.html')  # You need to create a sanky.html template

if __name__ == '__main__':
    app.run(debug=True)
