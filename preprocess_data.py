# preprocess_data.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# Create models folder if it doesn't exist
os.makedirs('models', exist_ok=True)

print("="*50)
print("STEP 1: Loading and Preprocessing Data")
print("="*50)

# Load the dataset
df = pd.read_csv('data/disease_data.csv')
print(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns")

# Create a copy for preprocessing
df_processed = df.copy()

# 1. Convert symptom columns from Yes/No to 1/0
symptom_columns = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']
for col in symptom_columns:
    df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0})
print("✅ Converted symptoms to 1/0")

# 2. Encode categorical columns
df_processed['Gender'] = df_processed['Gender'].map({'Male': 1, 'Female': 0})

bp_mapping = {'Low': 0, 'Normal': 1, 'High': 2}
df_processed['Blood Pressure'] = df_processed['Blood Pressure'].map(bp_mapping)

chol_mapping = {'Low': 0, 'Normal': 1, 'High': 2}
df_processed['Cholesterol Level'] = df_processed['Cholesterol Level'].map(chol_mapping)
print("✅ Encoded categorical variables")

# 3. Encode target variable (Outcome Variable)
target_mapping = {'Negative': 0, 'Positive': 1}
df_processed['Outcome Variable'] = df_processed['Outcome Variable'].map(target_mapping)

# 4. Encode Disease names
disease_encoder = LabelEncoder()
df_processed['Disease_Encoded'] = disease_encoder.fit_transform(df_processed['Disease'])
print("✅ Encoded disease names")

# 5. Select features for training
feature_columns = symptom_columns + ['Age', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'Disease_Encoded']
X = df_processed[feature_columns]
y = df_processed['Outcome Variable']

print(f"\n📊 Features shape: {X.shape}")
print(f"📊 Target shape: {y.shape}")
print(f"\n📋 Feature columns: {feature_columns}")

print(f"\n📈 Target distribution:")
print(y.value_counts())
print(f"   Positive: {(y==1).sum()} cases")
print(f"   Negative: {(y==0).sum()} cases")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Train set size: {len(X_train)}")
print(f"📊 Test set size: {len(X_test)}")

# Save preprocessed data
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

# Save encoders
joblib.dump(disease_encoder, 'models/disease_encoder.pkl')
joblib.dump(feature_columns, 'models/feature_columns.pkl')

print("\n💾 Saved preprocessed data and encoders!")

print("\n✅ Preprocessing complete! Ready for training.")