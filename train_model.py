# train_model.py - Complete Disease Prediction Model Training with Realistic Mapping
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create necessary folders
os.makedirs('models', exist_ok=True)
os.makedirs('screenshots', exist_ok=True)

print("="*60)
print("🏥 DISEASE PREDICTION MODEL TRAINING")
print("="*60)

# ============================================
# STEP 1: Load or Create Dataset with Realistic Mapping
# ============================================

dataset_path = 'data/disease_dataset.csv'

# Function for realistic disease assignment based on medical patterns
def assign_disease(row):
    """Assign disease based on realistic symptom patterns"""
    
    # Extract symptoms
    fever = row['Fever']
    cough = row['Cough']
    fatigue = row['Fatigue']
    difficulty_breathing = row['Difficulty_Breathing']
    headache = row['Headache']
    sore_throat = row['SoreThroat']
    body_ache = row['BodyAche']
    runny_nose = row['RunnyNose']
    
    # BRONCHITIS: Persistent cough + difficulty breathing + fatigue
    if cough and difficulty_breathing and fatigue:
        return 'Bronchitis'
    
    # INFLUENZA (Flu): Fever + body ache + fatigue + (cough or headache)
    if fever and body_ache and fatigue:
        if cough or headache:
            return 'Influenza'
    
    # COVID-19: Fever + cough + difficulty breathing + (fatigue or body ache)
    if fever and cough and difficulty_breathing:
        if fatigue or body_ache:
            return 'COVID-19'
    
    # COMMON COLD: Runny nose + sore throat + (cough or fatigue)
    if runny_nose and sore_throat:
        if cough or fatigue:
            return 'Common Cold'
    
    # ALLERGY: Runny nose + (no fever) + (sneezing pattern - fatigue or headache)
    if runny_nose and not fever:
        if fatigue or headache:
            return 'Allergy'
    
    # Secondary patterns
    
    # COVID-19 alternative: Loss of taste/smell (using headache + fatigue)
    if fever and headache and fatigue and not difficulty_breathing:
        return 'COVID-19'
    
    # Influenza alternative: High fever + severe body ache
    if fever and body_ache and not runny_nose:
        return 'Influenza'
    
    # Bronchitis alternative: Chronic cough + chest discomfort (fatigue)
    if cough and fatigue and not fever:
        return 'Bronchitis'
    
    # Common Cold alternative: Mild symptoms with runny nose
    if runny_nose and not fever and not difficulty_breathing:
        return 'Common Cold'
    
    # Default fallback - weighted random based on common patterns
    if fever:
        return np.random.choice(['Influenza', 'COVID-19'], p=[0.6, 0.4])
    elif cough and runny_nose:
        return np.random.choice(['Common Cold', 'Allergy'], p=[0.7, 0.3])
    elif difficulty_breathing:
        return np.random.choice(['Bronchitis', 'COVID-19'], p=[0.5, 0.5])
    else:
        return np.random.choice(['Common Cold', 'Allergy', 'Influenza'], p=[0.4, 0.4, 0.2])

if not os.path.exists(dataset_path):
    print("\n📝 Creating NEW dataset with REALISTIC symptom-disease mapping...")
    
    # Create dataset with 2000 samples for better training
    symptoms = ['Fever', 'Cough', 'Fatigue', 'Difficulty_Breathing', 
                'Headache', 'SoreThroat', 'BodyAche', 'RunnyNose']
    diseases = ['Influenza', 'Common Cold', 'COVID-19', 'Allergy', 'Bronchitis']
    
    data = []
    for _ in range(5000):  # Increased from 500 to 2000 samples
        row = {}
        
        # Generate realistic symptom combinations
        # Base symptoms probability
        row['Fever'] = np.random.choice([0, 1], p=[0.7, 0.3])
        row['Cough'] = np.random.choice([0, 1], p=[0.6, 0.4])
        row['Fatigue'] = np.random.choice([0, 1], p=[0.65, 0.35])
        row['Difficulty_Breathing'] = np.random.choice([0, 1], p=[0.85, 0.15])
        row['Headache'] = np.random.choice([0, 1], p=[0.7, 0.3])
        row['SoreThroat'] = np.random.choice([0, 1], p=[0.75, 0.25])
        row['BodyAche'] = np.random.choice([0, 1], p=[0.8, 0.2])
        row['RunnyNose'] = np.random.choice([0, 1], p=[0.7, 0.3])
        
        # Age distribution (more realistic)
        row['Age'] = np.random.choice(
            list(range(1, 100)),
            p=[0.02] * 20 + [0.015] * 30 + [0.01] * 30 + [0.005] * 19  # More young people
        )
        
        row['Gender'] = np.random.choice([0, 1], p=[0.5, 0.5])
        row['Blood_Pressure'] = np.random.choice([0, 1, 2], p=[0.2, 0.6, 0.2])
        row['Cholesterol_Level'] = np.random.choice([0, 1, 2], p=[0.2, 0.6, 0.2])
        
        # Assign disease using realistic mapping
        row['Disease'] = assign_disease(row)
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(dataset_path, index=False)
    print(f"✅ Dataset created with {len(df)} samples!")
    print(f"   (Using realistic medical symptom-disease patterns)")
else:
    df = pd.read_csv(dataset_path)
    print(f"✅ Dataset loaded: {len(df)} samples")

print(f"\n📊 Dataset columns: {df.columns.tolist()}")
print(f"\n📋 Disease distribution:")
disease_counts = df['Disease'].value_counts()
for disease, count in disease_counts.items():
    print(f"   {disease}: {count} samples ({count/len(df)*100:.1f}%)")

# ============================================
# STEP 2: Prepare Features and Target
# ============================================

print("\n" + "="*60)
print("STEP 2: Preparing Features & Target")
print("="*60)

# All 8 symptoms + health metrics
feature_columns = [
    'Fever', 'Cough', 'Fatigue', 'Difficulty_Breathing',
    'Headache', 'SoreThroat', 'BodyAche', 'RunnyNose',
    'Age', 'Gender', 'Blood_Pressure', 'Cholesterol_Level'
]

# Separate features and target
X = df[feature_columns]
y = df['Disease']

print(f"✅ Features shape: {X.shape}")
print(f"✅ Features: {list(X.columns)}")
print(f"✅ Target classes: {y.unique().tolist()}")

# ============================================
# STEP 3: Encode Target Labels
# ============================================

print("\n" + "="*60)
print("STEP 3: Encoding Disease Names")
print("="*60)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save disease mapping
disease_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
print(f"✅ Disease Mapping:")
for disease, code in disease_mapping.items():
    print(f"   {code} → {disease}")

# ============================================
# STEP 4: Train-Test Split
# ============================================

print("\n" + "="*60)
print("STEP 4: Train-Test Split")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"✅ Training samples: {len(X_train)}")
print(f"✅ Test samples: {len(X_test)}")

# ============================================
# STEP 5: Train Random Forest Model
# ============================================

print("\n" + "="*60)
print("STEP 5: Training Random Forest Classifier")
print("="*60)

rf_model = RandomForestClassifier(
    n_estimators=150,  # Increased for better accuracy
    max_depth=15,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'
)

print("🔄 Training model...")
rf_model.fit(X_train, y_train)
print("✅ Model training completed!")

# ============================================
# STEP 6: Model Evaluation
# ============================================

print("\n" + "="*60)
print("STEP 6: Model Evaluation")
print("="*60)

# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 Model Performance:")
print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Classification report
print(f"\n📋 Detailed Classification Report:")
class_names = label_encoder.classes_
print(classification_report(y_test, y_pred, target_names=class_names))

# ============================================
# STEP 7: Feature Importance
# ============================================

print("\n" + "="*60)
print("STEP 7: Feature Importance Analysis")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔝 Top 10 Most Important Features:")
for i, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

# ============================================
# STEP 8: Save Visualizations
# ============================================

print("\n" + "="*60)
print("STEP 8: Saving Visualizations")
print("="*60)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Confusion Matrix - Disease Prediction', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Disease', fontsize=12)
plt.ylabel('Actual Disease', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('screenshots/confusion_matrix.png', dpi=100, bbox_inches='tight')
plt.close()
print("📸 Confusion matrix saved to 'screenshots/confusion_matrix.png'")

# Plot feature importance
plt.figure(figsize=(12, 8))
colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(feature_importance.head(12))))
plt.barh(feature_importance.head(12)['feature'], 
         feature_importance.head(12)['importance'],
         color=colors)
plt.xlabel('Importance Score', fontsize=12)
plt.title('Feature Importances for Disease Prediction', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('screenshots/feature_importance.png', dpi=100, bbox_inches='tight')
plt.close()
print("📸 Feature importance plot saved to 'screenshots/feature_importance.png'")

# ============================================
# STEP 9: Save Model and Encoders
# ============================================

print("\n" + "="*60)
print("STEP 9: Saving Model & Encoders")
print("="*60)

# Save model
joblib.dump(rf_model, 'models/disease_model.pkl')
print("✅ Model saved to 'models/disease_model.pkl'")

# Save feature columns
joblib.dump(feature_columns, 'models/feature_columns.pkl')
print("✅ Feature columns saved to 'models/feature_columns.pkl'")

# Save label encoder
joblib.dump(label_encoder, 'models/disease_encoder.pkl')
print("✅ Disease encoder saved to 'models/disease_encoder.pkl'")

# Save disease mapping for reference
import json
with open('models/disease_mapping.json', 'w') as f:
    json.dump(disease_mapping, f, indent=4)
print("✅ Disease mapping saved to 'models/disease_mapping.json'")

# ============================================
# STEP 10: Summary
# ============================================

print("\n" + "="*60)
print("🎉 TRAINING COMPLETE!")
print("="*60)
print(f"""
📊 Final Summary:
   • Dataset Size: {len(df)} samples
   • Features: {len(feature_columns)}
   • Disease Classes: {len(class_names)}
   • Model Accuracy: {accuracy:.2%}
   • Model Saved: models/disease_model.pkl

🏥 Diseases that can be predicted:
   {', '.join(class_names)}

📋 Symptom-Disease Mapping Rules:
   • BRONCHITIS: Cough + Difficulty Breathing + Fatigue
   • INFLUENZA: Fever + Body Ache + Fatigue + (Cough/Headache)
   • COVID-19: Fever + Cough + Difficulty Breathing
   • COMMON COLD: Runny Nose + Sore Throat + (Cough/Fatigue)
   • ALLERGY: Runny Nose + No Fever

💾 Saved Files:
   • models/disease_model.pkl
   • models/feature_columns.pkl
   • models/disease_encoder.pkl
   • models/disease_mapping.json
   • screenshots/confusion_matrix.png
   • screenshots/feature_importance.png

🚀 Next Step: Run 'python app.py' to start the web application!
""")

# Test prediction example
print("\n" + "="*60)
print("🔍 TEST PREDICTION EXAMPLES")
print("="*60)

test_cases = [
    {"name": "Bronchitis", "symptoms": {"Cough": 1, "Difficulty_Breathing": 1, "Fatigue": 1}},
    {"name": "Influenza", "symptoms": {"Fever": 1, "BodyAche": 1, "Fatigue": 1, "Cough": 1}},
    {"name": "COVID-19", "symptoms": {"Fever": 1, "Cough": 1, "Difficulty_Breathing": 1}},
    {"name": "Common Cold", "symptoms": {"RunnyNose": 1, "SoreThroat": 1, "Cough": 1}},
    {"name": "Allergy", "symptoms": {"RunnyNose": 1, "Fatigue": 1}}
]

for test in test_cases:
    # Create feature vector
    features = {col: 0 for col in feature_columns}
    features.update(test["symptoms"])
    features['Age'] = 30
    features['Gender'] = 0
    features['Blood_Pressure'] = 1
    features['Cholesterol_Level'] = 1
    
    X_test_sample = np.array([[features[col] for col in feature_columns]])
    pred = rf_model.predict(X_test_sample)[0]
    disease = label_encoder.inverse_transform([pred])[0]
    prob = np.max(rf_model.predict_proba(X_test_sample)[0])
    
    print(f"\n✅ {test['name']} symptoms → Predicted: {disease} ({prob*100:.1f}% confidence)")