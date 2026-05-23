# check_dataset.py
import pandas as pd
import os

# Check if file exists in data folder
file_path = 'data/disease_data.csv'

if os.path.exists(file_path):
    print("✅ File found in data folder!")
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    print("\n" + "="*50)
    print("DATASET INFORMATION")
    print("="*50)
    
    print(f"\n📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print(f"\n📝 Column Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i}. {col}")
    
    print(f"\n👀 First 3 rows:")
    print(df.head(3))
    
    print(f"\n📈 Data Types:")
    print(df.dtypes)
    
    print(f"\n🔍 Missing Values:")
    print(df.isnull().sum())
    
    # Check the last column (likely the disease column)
    disease_col = df.columns[-1]
    print(f"\n🎯 Target/Disease Column: '{disease_col}'")
    print(f"\n📊 Disease Distribution:")
    print(df[disease_col].value_counts())
    
    # Check what values are in symptom columns (True/False or 1/0)
    print(f"\n✅ Sample of first 5 rows (symptoms only):")
    print(df.iloc[:5, :-1])  # Show all except last column
    
else:
    print("❌ File not found in 'data/disease_data.csv'")
    print(f"Current file location: {file_path}")
    print("\nPlease copy the CSV file to the 'data/' folder")