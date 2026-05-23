# app.py - PHASE 2 Enhanced Version
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from chatbot import chatbot
import uuid

app = Flask(__name__)

# Load model and encoders
print("Loading model and encoders...")
model = joblib.load('models/disease_model.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')
disease_encoder = joblib.load('models/disease_encoder.pkl')
print("✅ Model loaded successfully!")

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
        # Get data from request
        data = request.get_json()
        
        # Extract patient data
        patient_data = {
            'Fever': int(data.get('Fever', 0)),
            'Cough': int(data.get('Cough', 0)),
            'Fatigue': int(data.get('Fatigue', 0)),
            'Difficulty Breathing': int(data.get('Difficulty_Breathing', 0)),
            'Age': int(data.get('Age', 30)),
            'Gender': int(data.get('Gender', 0)),
            'Blood Pressure': int(data.get('Blood_Pressure', 1)),
            'Cholesterol Level': int(data.get('Cholesterol_Level', 1)),
            'Disease_Encoded': 0
        }
        
        # Create feature vector
        features = np.array([[patient_data[col] for col in feature_columns]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Get confidence
        confidence = round(max(probabilities) * 100, 2)
        result = "Positive" if prediction == 1 else "Negative"
        
        # Get enhanced severity analysis
        severity, severity_icon, risk_factors, action_required = get_severity_enhanced(patient_data, confidence, result)
        
        # Get specialist recommendations
        specialists = specialist_mapping[result][severity]
        
        # Get symptom-specific advice
        symptom_advice = []
        symptoms_list = []
        if patient_data['Fever']:
            symptom_advice.extend(medical_advice_detailed['Fever'])
            symptoms_list.append('Fever')
        if patient_data['Cough']:
            symptom_advice.extend(medical_advice_detailed['Cough'])
            symptoms_list.append('Cough')
        if patient_data['Fatigue']:
            symptom_advice.extend(medical_advice_detailed['Fatigue'])
            symptoms_list.append('Fatigue')
        if patient_data['Difficulty Breathing']:
            symptom_advice.extend(medical_advice_detailed['Difficulty Breathing'])
            symptoms_list.append('Difficulty Breathing')
        
        # Remove duplicates from symptom advice
        symptom_advice = list(dict.fromkeys(symptom_advice))
        
        # Get lifestyle recommendations based on age and risk
        age = patient_data['Age']
        if age > 50 or severity == "Severe":
            lifestyle_recs = lifestyle_advice['diet'] + lifestyle_advice['exercise']
        else:
            lifestyle_recs = lifestyle_advice['diet'][:2] + lifestyle_advice['exercise'][:2] + lifestyle_advice['prevention'][:2]
        
        # Prepare response
        response = {
            'prediction': result,
            'confidence': confidence,
            'probability_negative': round(probabilities[0] * 100, 2),
            'probability_positive': round(probabilities[1] * 100, 2),
            'severity': severity,
            'severity_icon': severity_icon,
            'risk_factors': risk_factors,
            'action_required': action_required,
            'specialists': specialists,
            'symptom_advice': symptom_advice[:5],  # Top 5 advice items
            'lifestyle_advice': lifestyle_recs[:5],  # Top 5 lifestyle tips
            'symptoms_count': len(symptoms_list),
            'symptoms_list': symptoms_list,
            'age': patient_data['Age'],
            'gender': 'Male' if patient_data['Gender'] == 1 else 'Female',
            'bp_status': ['Low', 'Normal', 'High'][patient_data['Blood Pressure']],
            'cholesterol_status': ['Low', 'Normal', 'High'][patient_data['Cholesterol Level']],
            'disclaimer': "⚠️ DISCLAIMER: This is an AI prediction system based on statistical analysis. This is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition."
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

# Add this route after your existing routes

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