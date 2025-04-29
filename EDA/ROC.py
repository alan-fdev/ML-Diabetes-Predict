import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# ===== Load model & scaler =====
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# ===== Load data & preprocessing =====
data = pd.read_csv('../Datasheet/diabetes.csv')

cols_with_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for col in cols_with_zero:
    data[col] = data[col].replace(0, pd.NA)
    data[col] = data[col].fillna(data[col].median())

X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_scaled = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ===== Prediksi probabilitas =====
y_prob = model.predict_proba(X_test)[:, 1]

# ===== Hitung ROC & AUC =====
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# ===== Plot dan Simpan Gambar =====
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()

# Simpan ke file
plt.savefig('roc_curve_logreg.png')  # Gambar disimpan di direktori kerja saat ini
plt.close()
