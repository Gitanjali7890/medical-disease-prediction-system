# app.py - PHASE 2 Enhanced Version
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from chatbot import chatbot
import uuid
# Add these imports at the top of app.py
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from functools import wraps
from database import create_user, authenticate_user, save_prediction, get_user_predictions


app = Flask(__name__)

# Add secret key (change this in production)
app.secret_key = 'cura-health-secret-key-2024'
# Load model and encoders
print("Loading model and encoders...")
model = joblib.load('models/disease_model.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')
disease_encoder = joblib.load('models/disease_encoder.pkl')
print("✅ Model loaded successfully!")

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# Enhanced Disease to Specialist Mapping
specialist_mapping = {
    'Positive': {
        'Severe': [
            "🏥 Emergency Medicine Specialist - Immediate care needed",
            "🫁 Pulmonologist - For respiratory symptoms",
            "🩺 Infectious Disease Specialist",
            "💊 Clinical Pharmacist"
        ],
        'Moderate': [
            "👨‍⚕️ General Physician - Primary consultation",
            "🫁 Pulmonologist - If breathing difficulties present",
            "🩺 Internal Medicine Specialist",
            "📋 Schedule within 24-48 hours"
        ],
        'Mild': [
            "👨‍⚕️ General Practitioner - Initial assessment",
            "💊 Pharmacist - For medication guidance",
            "📞 Telehealth consultation available"
        ]
    },
    'Negative': {
        'Severe': [
            "👨‍⚕️ General Physician - Comprehensive checkup",
            "❤️ Cardiologist - Heart health evaluation",
            "🩸 Pathologist - Lab tests recommended"
        ],
        'Moderate': [
            "👨‍⚕️ Family Medicine Doctor",
            "🥗 Nutritionist - Preventive care",
            "🏃‍♂️ Physical Therapist - Exercise guidance"
        ],
        'Mild': [
            "👨‍⚕️ General Practitioner - Routine checkup",
            "🥗 Dietitian - For preventive health advice",
            "💪 Wellness Coach - Lifestyle optimization"
        ]
    }
}

# Detailed Medical Advice based on symptoms
medical_advice_detailed = {
    'Fever': [
        '🌡️ Monitor body temperature every 4-6 hours',
        '💧 Stay hydrated - drink at least 8-10 glasses of water daily',
        '🛁 Take lukewarm baths to reduce fever',
        '💊 Use over-the-counter fever reducers if temperature exceeds 101°F',
        '🛌 Get adequate rest to help immune system'
    ],
    'Cough': [
        '🍯 Use honey and ginger tea for natural relief',
        '❄️ Avoid cold foods and drinks',
        '💨 Use a humidifier to moisten the air',
        '🩺 Consult doctor if cough persists for more than 7 days',
        '😷 Wear mask to prevent spreading'
    ],
    'Fatigue': [
        '😴 Ensure 7-8 hours of quality sleep',
        '🥬 Eat iron-rich foods like spinach and legumes',
        '☕ Take short breaks throughout the day',
        '🚶 Light exercise like walking can help boost energy',
        '💆 Practice stress reduction techniques'
    ],
    'Difficulty Breathing': [
        '🚨 Seek immediate medical attention if severe',
        '💺 Sit upright to ease breathing',
        '💨 Use prescribed inhalers if available',
        '🧘 Practice deep breathing exercises',
        '📞 Call emergency services if condition worsens'
    ]
}

# Lifestyle recommendations
lifestyle_advice = {
    'diet': [
        '🥗 Eat a balanced diet rich in fruits and vegetables',
        '🍚 Reduce processed foods and sugar intake',
        '🥛 Include probiotics for gut health',
        '💧 Stay hydrated throughout the day',
        '🥩 Choose lean proteins over red meat'
    ],
    'exercise': [
        '🏃 30 minutes of moderate exercise daily',
        '💪 Include both cardio and strength training',
        '🧘 Practice yoga or meditation for stress relief',
        '🚶 Take regular walks, especially after meals',
        '🏊 Low-impact exercises like swimming are great'
    ],
    'prevention': [
        '🧼 Wash hands frequently with soap',
        '📅 Get annual health checkups',
        '💉 Maintain proper vaccination schedule',
        '🚭 Avoid smoking and limit alcohol consumption',
        '😷 Wear masks in crowded places during flu season'
    ]
}

def get_severity_enhanced(patient_data, confidence, prediction):
    """Enhanced severity detection with detailed analysis"""
    severity_score = 0
    risk_factors = []
    
    # Check symptoms
    symptoms_count = sum([
        patient_data.get('Fever', 0),
        patient_data.get('Cough', 0),
        patient_data.get('Fatigue', 0),
        patient_data.get('Difficulty Breathing', 0)
    ])
    
    if symptoms_count >= 3:
        severity_score += 3
        risk_factors.append("⚠️ Multiple symptoms present (3 or more)")
    elif symptoms_count >= 2:
        severity_score += 2
        risk_factors.append("⚠️ Several symptoms detected")
    elif symptoms_count >= 1:
        severity_score += 1
        risk_factors.append("📋 Symptoms present")
    
    # Check age risk
    age = patient_data.get('Age', 0)
    if age >= 60:
        severity_score += 3
        risk_factors.append("👴 Senior age group (60+) - Higher risk category")
    elif age >= 50:
        severity_score += 2
        risk_factors.append("🧑 Middle age (50-59) - Moderate risk")
    elif age <= 5:
        severity_score += 2
        risk_factors.append("👶 Young child - Vulnerable group")
    elif age >= 65:
        severity_score += 1
        risk_factors.append("👵 Elderly - Requires extra care")
    
    # Check blood pressure
    bp = patient_data.get('Blood Pressure', 1)
    if bp == 2:  # High
        severity_score += 2
        risk_factors.append("❤️ High blood pressure detected - Cardiovascular risk")
    elif bp == 0:  # Low
        severity_score += 1
        risk_factors.append("💙 Low blood pressure - Monitor regularly")
    
    # Check cholesterol
    cholesterol = patient_data.get('Cholesterol Level', 1)
    if cholesterol == 2:  # High
        severity_score += 2
        risk_factors.append("🩸 High cholesterol levels - Diet modification needed")
    elif cholesterol == 0:  # Low
        severity_score += 1
        risk_factors.append("📉 Low cholesterol - May need nutritional assessment")
    
    # Check prediction confidence and outcome
    if prediction == "Positive":
        severity_score += 2
        risk_factors.append("🎯 Positive outcome predicted")
        if confidence > 85:
            severity_score += 2
            risk_factors.append("📊 High prediction confidence (>85%)")
        elif confidence > 70:
            severity_score += 1
            risk_factors.append("📈 Moderate prediction confidence")
    else:
        if confidence > 85:
            risk_factors.append("✅ High confidence in negative outcome")
    
    # Determine severity level
    if severity_score >= 8:
        return "Severe", "🔴", risk_factors, "🚨 IMMEDIATE ACTION REQUIRED: Please seek medical attention within 24 hours"
    elif severity_score >= 5:
        return "Moderate", "🟠", risk_factors, "⚠️ MEDICAL ATTENTION RECOMMENDED: Schedule a doctor's appointment within 2-3 days"
    else:
        return "Mild", "🟢", risk_factors, "✅ LOW RISK: Monitor symptoms and maintain healthy lifestyle"

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for disease prediction"""
    try:
        data = request.get_json()
        print("Received data:", data)  # Debug print
        
        # Extract patient data - MATCH EXACTLY with feature_columns
        patient_data = {
            'Fever': int(data.get('Fever', 0)),
            'Cough': int(data.get('Cough', 0)),
            'Fatigue': int(data.get('Fatigue', 0)),
            'Difficulty_Breathing': int(data.get('Difficulty_Breathing', 0)),
            'Headache': int(data.get('Headache', 0)),
            'SoreThroat': int(data.get('SoreThroat', 0)),
            'BodyAche': int(data.get('BodyAche', 0)),
            'RunnyNose': int(data.get('RunnyNose', 0)),
            'Age': int(data.get('Age', 30)),
            'Gender': int(data.get('Gender', 0)),
            'Blood_Pressure': int(data.get('Blood_Pressure', 1)),
            'Cholesterol_Level': int(data.get('Cholesterol_Level', 1))
        }
        
        print("Patient data dict:", patient_data)  # Debug print
        
        # Create feature vector in correct order
        feature_columns = joblib.load('models/feature_columns.pkl')
        print("Feature columns from model:", feature_columns)  # Debug print
        
        features = np.array([[patient_data[col] for col in feature_columns]])
        print("Features array:", features)  # Debug print
        
        # Make prediction
        prediction_encoded = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Get actual disease name
        predicted_disease = disease_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get confidence
        confidence = round(max(probabilities) * 100, 2)
        
        # Get top 3 predictions
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_diseases = []
        for idx in top_3_indices:
            top_3_diseases.append({
                'disease': disease_encoder.inverse_transform([idx])[0],
                'probability': round(probabilities[idx] * 100, 2)
            })
        
        # Count symptoms (all 8)
        symptoms_count = sum([
            patient_data['Fever'], patient_data['Cough'], patient_data['Fatigue'],
            patient_data['Difficulty_Breathing'], patient_data['Headache'],
            patient_data['SoreThroat'], patient_data['BodyAche'], patient_data['RunnyNose']
        ])
        
        # Get list of selected symptoms for saving to history
        selected_symptoms_list = []
        symptom_names = ['Fever', 'Cough', 'Fatigue', 'Difficulty_Breathing', 'Headache', 'SoreThroat', 'BodyAche', 'RunnyNose']
        for symptom in symptom_names:
            if patient_data.get(symptom, 0) == 1:
                # Convert Difficulty_Breathing to readable format
                if symptom == 'Difficulty_Breathing':
                    selected_symptoms_list.append('Difficulty Breathing')
                else:
                    selected_symptoms_list.append(symptom)
        symptoms_str = ', '.join(selected_symptoms_list) if selected_symptoms_list else 'None'
        
        # Determine severity
        if symptoms_count >= 5 or predicted_disease in ['COVID-19', 'Influenza']:
            severity = "Moderate"
            severity_icon = "🟠"
            action = "⚠️ MEDICAL ATTENTION RECOMMENDED: Schedule a doctor's appointment within 2-3 days"
        elif symptoms_count >= 3:
            severity = "Mild"
            severity_icon = "🟢"
            action = "✅ LOW RISK: Monitor symptoms and maintain healthy lifestyle"
        else:
            severity = "Minimal"
            severity_icon = "🔵"
            action = "✅ Very low risk. Maintain healthy habits."
        
        # Specialist mapping
        specialist_map = {
            'Influenza': ['👨‍⚕️ General Physician', '🫁 Pulmonologist', '💊 Pharmacist'],
            'COVID-19': ['🫁 Pulmonologist', '🩺 Infectious Disease Specialist', '👨‍⚕️ General Physician'],
            'Common Cold': ['👨‍⚕️ General Practitioner', '💊 Pharmacist', '📞 Telehealth'],
            'Allergy': ['🩺 Allergist', '👨‍⚕️ General Physician', '🥗 Nutritionist'],
            'Bronchitis': ['🫁 Pulmonologist', '👨‍⚕️ General Physician', '💊 Respiratory Therapist']
        }
        specialists = specialist_map.get(predicted_disease, ['👨‍⚕️ General Physician', '📞 Telehealth consultation'])
        specialists_str = ', '.join(specialists)
        
        # ============================================
        # SAVE PREDICTION TO HISTORY (IF USER LOGGED IN)
        # ============================================
        if 'user_id' in session:
            try:
                from database import save_prediction
                save_prediction(
                    session['user_id'],
                    symptoms_str,
                    predicted_disease,
                    confidence,
                    severity,
                    specialists_str
                )
                print(f"✅ Prediction saved for user: {session.get('username', 'Unknown')}")
            except Exception as e:
                print(f"⚠️ Could not save prediction: {e}")
        
        response = {
            'predicted_disease': predicted_disease,
            'confidence': confidence,
            'top_3_predictions': top_3_diseases,
            'severity': severity,
            'severity_icon': severity_icon,
            'action_required': action,
            'specialists': specialists,
            'symptoms_count': symptoms_count,
            'age': patient_data['Age'],
            'gender': 'Male' if patient_data['Gender'] == 1 else 'Female',
            'bp_status': ['Low', 'Normal', 'High'][patient_data['Blood_Pressure']],
            'cholesterol_status': ['Low', 'Normal', 'High'][patient_data['Cholesterol_Level']],
            'disclaimer': "⚠️ DISCLAIMER: This is an AI prediction system based on statistical analysis. This is not a substitute for professional medical advice."
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    # Add this route after your existing routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        result = authenticate_user(username, password)
        
        if result['success']:
            session['user_id'] = result['user_id']
            session['username'] = result['username']
            session['email'] = result['email']
            flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash(result['error'], 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters', 'error')
            return render_template('register.html')
        
        result = create_user(username, email, password)
        
        if result['success']:
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        else:
            flash(result['error'], 'error')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    predictions = get_user_predictions(session['user_id'])
    
    unique_diseases = len(set(p['disease'] for p in predictions)) if predictions else 0
    avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions) if predictions else 0
    
    # Calculate severity counts for dashboard
    severity_counts = {'Mild': 0, 'Moderate': 0, 'Severe': 0, 'Minimal': 0}
    for pred in predictions:
        severity = pred.get('severity', 'Mild')
        if severity in severity_counts:
            severity_counts[severity] += 1
    
    return render_template('dashboard.html', 
                         predictions=predictions,
                         unique_diseases=unique_diseases,
                         avg_confidence=avg_confidence,
                         severity_counts=severity_counts)
# Add this import at the top
from nlp_processor import nlp_processor

# Add this route after your existing routes
@app.route('/api/nlp_process', methods=['POST'])
def nlp_process():
    """Process natural language symptom input"""
    try:
        data = request.get_json()
        user_text = data.get('text', '')
        
        if not user_text.strip():
            return jsonify({
                'success': False,
                'error': 'Please enter your symptoms'
            }), 400
        
        # Process the text
        result = nlp_processor.process_message(user_text)
        
        # Convert detected symptoms to form data format
        form_data = {}
        for symptom in result['detected_symptoms']:
            form_data[symptom] = 1
        
        return jsonify({
            'success': True,
            'detected_symptoms': result['detected_symptoms'],
            'unknown_symptoms': result['unknown_symptoms'],
            'warnings': result['warnings'],
            'suggestions': result['suggestions'],
            'has_known': result['has_known'],
            'has_unknown': result['has_unknown'],
            'form_data': form_data
        })
    
    except Exception as e:
        print(f"NLP Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
@app.route('/api/chat', methods=['POST'])
def chat():
    """Chatbot API endpoint - maintains conversation"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        # Process message with chatbot (now maintains conversation)
        response = chatbot.process_message(user_message, session_id)
        detected_symptoms = chatbot.get_detected_symptoms(session_id)
        
        return jsonify({
            'success': True,
            'response': response,
            'session_id': session_id,
            'detected_symptoms': detected_symptoms,
            'has_symptoms': len(detected_symptoms) > 0
        })
    
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    """Clear chatbot session"""
    try:
        data = request.get_json()
        session_id = data.get('session_id', '')
        if session_id:
            chatbot.clear_session(session_id)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/apply_symptoms', methods=['POST'])
def apply_symptoms():
    """Apply chatbot detected symptoms to the main form"""
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', [])
        
        # Map symptom names to form field names
        symptom_map = {
            'fever': 'Fever',
            'cough': 'Cough',
            'fatigue': 'Fatigue',
            'difficulty_breathing': 'Difficulty_Breathing',
            'headache': 'Headache',
            'sore_throat': 'SoreThroat',
            'body_ache': 'BodyAche',
            'nausea': 'Nausea',
            'runny_nose': 'RunnyNose'
        }
        
        form_data = {}
        for symptom in symptoms:
            if symptom in symptom_map:
                form_data[symptom_map[symptom]] = 1
        
        return jsonify({
            'success': True,
            'form_data': form_data
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)