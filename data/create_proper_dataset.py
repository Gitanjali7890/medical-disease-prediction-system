import pandas as pd
import numpy as np

# Create proper dataset with 8 symptoms
data = {
    'Fever': [1,0,1,0,1,0,1,0,1,0,1,0,1,1,0,0],
    'Cough': [1,1,0,1,0,1,1,0,1,1,0,1,0,1,0,1],
    'Fatigue': [1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0],
    'Difficulty_Breathing': [0,0,1,0,1,0,1,0,0,1,0,1,1,0,1,0],
    'Headache': [1,0,1,0,1,0,0,1,1,0,1,0,1,1,0,1],
    'SoreThroat': [0,1,0,1,0,1,1,0,0,1,0,1,1,0,1,0],
    'BodyAche': [1,1,0,1,0,1,0,1,1,0,1,1,0,1,0,0],
    'RunnyNose': [0,0,1,0,1,0,1,0,0,1,1,0,1,0,1,1],
    'Age': [25,30,45,22,35,50,28,65,40,19,55,32,48,60,27,38],
    'Gender': [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
    'Blood_Pressure': [1,1,2,1,0,1,2,1,1,2,1,0,1,1,2,1],
    'Cholesterol_Level': [1,1,2,1,1,2,1,1,0,1,1,2,1,1,2,1],
    'Disease': [
        'Influenza', 'Common Cold', 'COVID-19', 'Common Cold',
        'Influenza', 'Allergy', 'COVID-19', 'Common Cold',
        'Influenza', 'Allergy', 'COVID-19', 'Common Cold',
        'Influenza', 'Bronchitis', 'COVID-19', 'Allergy'
    ]
}

df = pd.DataFrame(data)

# Encode Disease to numbers
disease_mapping = {
    'Influenza': 0,
    'Common Cold': 1, 
    'COVID-19': 2,
    'Allergy': 3,
    'Bronchitis': 4
}
df['Disease_Encoded'] = df['Disease'].map(disease_mapping)

# Save dataset
df.to_csv('data/disease_dataset.csv', index=False)
print("✅ Dataset created with 8 symptoms!")
print(f"\nDisease Mapping: {disease_mapping}")
print(f"\nDataset shape: {df.shape}")
print(df.head())