# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create screenshots folder if it doesn't exist
os.makedirs('screenshots', exist_ok=True)

print("="*50)
print("STEP 2: Training Machine Learning Model")
print("="*50)

# Load preprocessed data
X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').values.ravel()
y_test = pd.read_csv('data/y_test.csv').values.ravel()

print(f"✅ Loaded training data: {len(X_train)} samples")
print(f"✅ Loaded test data: {len(X_test)} samples")
print(f"\n📋 Features: {list(X_train.columns)}")

# Train Random Forest model
print("\n🔄 Training Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)
print("✅ Model training completed!")

# Make predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_proba)

print(f"\n📊 Model Performance:")
print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   AUC Score: {auc_score:.4f}")

# Classification report
print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔝 Top 10 Most Important Features:")
for i, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix - Disease Outcome Prediction', fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('screenshots/confusion_matrix.png', dpi=100)
print("\n📸 Confusion matrix saved to 'screenshots/confusion_matrix.png'")

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance.head(10)['feature'], 
         feature_importance.head(10)['importance'])
plt.xlabel('Importance Score', fontsize=12)
plt.title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('screenshots/feature_importance.png', dpi=100)
print("📸 Feature importance plot saved to 'screenshots/feature_importance.png'")

# Save model
joblib.dump(rf_model, 'models/disease_model.pkl')
print("\n💾 Model saved to 'models/disease_model.pkl'")

# Save metrics
with open('models/model_metrics.txt', 'w') as f:
    f.write(f"Model: Random Forest Classifier\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"AUC Score: {auc_score:.4f}\n")
    f.write(f"Features: {', '.join(X_train.columns)}\n")

print("\n🎉 Training complete! Model is ready for predictions.")