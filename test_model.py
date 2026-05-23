# test_model.py
import joblib
import numpy as np
import pandas as pd

print("="*50)
print("TESTING MODEL WITH SAMPLE PATIENTS")
print("="*50)

# Load model and encoders
model = joblib.load('models/disease_model.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')
disease_encoder = joblib.load('models/disease_encoder.pkl')

print(f"\n✅ Model loaded successfully!")
print(f"✅ Features: {feature_columns}")
print(f"✅ Available diseases: {list(disease_encoder.classes_)}")

# Test Case 1: High risk patient (elderly with symptoms)
patient1 = {
    'Fever': 1,
    'Cough': 1,
    'Fatigue': 1,
    'Difficulty Breathing': 1,
    'Age': 70,
    'Gender': 1,  # Male
    'Blood Pressure': 2,  # High
    'Cholesterol Level': 2,  # High
    'Disease_Encoded': 0  # Influenza
}

# Test Case 2: Low risk patient (young with no symptoms)
patient2 = {
    'Fever': 0,
    'Cough': 0,
    'Fatigue': 0,
    'Difficulty Breathing': 0,
    'Age': 25,
    'Gender': 0,  # Female
    'Blood Pressure': 1,  # Normal
    'Cholesterol Level': 1,  # Normal
    'Disease_Encoded': 0  # Influenza
}

# Test Case 3: Moderate risk patient
patient3 = {
    'Fever': 1,
    'Cough': 0,
    'Fatigue': 1,
    'Difficulty Breathing': 0,
    'Age': 45,
    'Gender': 1,
    'Blood Pressure': 1,
    'Cholesterol Level': 2,
    'Disease_Encoded': 1  # Common Cold
}

# Create feature vectors
features1 = np.array([[patient1[col] for col in feature_columns]])
features2 = np.array([[patient2[col] for col in feature_columns]])
features3 = np.array([[patient3[col] for col in feature_columns]])

# Predict
pred1 = model.predict(features1)[0]
prob1 = model.predict_proba(features1)[0]
pred2 = model.predict(features2)[0]
prob2 = model.predict_proba(features2)[0]
pred3 = model.predict(features3)[0]
prob3 = model.predict_proba(features3)[0]

print("\n" + "="*50)
print("PREDICTION RESULTS")
print("="*50)

print("\n🏥 PATIENT 1 (High Risk - Elderly with multiple symptoms):")
print(f"   - Age: 70, Symptoms: Fever, Cough, Fatigue, Breathing difficulty")
print(f"   - Blood Pressure: High, Cholesterol: High")
print(f"   📊 Prediction: {'⚠️ POSITIVE (High Risk)' if pred1 == 1 else '✅ NEGATIVE (Low Risk)'}")
print(f"   📈 Confidence: {max(prob1)*100:.2f}%")
print(f"   📉 Negative Probability: {prob1[0]*100:.2f}%")
print(f"   📈 Positive Probability: {prob1[1]*100:.2f}%")

print("\n🏥 PATIENT 2 (Low Risk - Young with no symptoms):")
print(f"   - Age: 25, Symptoms: None")
print(f"   - Blood Pressure: Normal, Cholesterol: Normal")
print(f"   📊 Prediction: {'⚠️ POSITIVE' if pred2 == 1 else '✅ NEGATIVE'}")
print(f"   📈 Confidence: {max(prob2)*100:.2f}%")

print("\n🏥 PATIENT 3 (Moderate Risk - Middle-aged with some symptoms):")
print(f"   - Age: 45, Symptoms: Fever, Fatigue")
print(f"   - Blood Pressure: Normal, Cholesterol: High")
print(f"   📊 Prediction: {'⚠️ POSITIVE' if pred3 == 1 else '✅ NEGATIVE'}")
print(f"   📈 Confidence: {max(prob3)*100:.2f}%")

print("\n" + "="*50)
print("✅ Model test completed successfully!")
print("="*50)

# Additional info
print("\n📊 Model Information:")
print(f"   - Algorithm: Random Forest Classifier")
print(f"   - Number of features: {len(feature_columns)}")
print(f"   - Number of disease classes: {len(disease_encoder.classes_)}")