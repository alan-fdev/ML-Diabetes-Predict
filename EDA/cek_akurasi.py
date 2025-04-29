import pandas as pd 
import joblib

from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split

# === Load model dan scaler ===
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# === Load ulang data dan preprocessing ===
data = pd.read_csv('../Datasheet/diabetes.csv')

# Bersihkan data
cols_with_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for col in cols_with_zero:
    data[col] = data[col].replace(0, pd.NA)
    data[col] = data[col].fillna(data[col].median())

# pisahkan fitur dan target 
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Normalisasi dengan scaler yang sudah dilatih 
X_scaled = scaler.transform(X)

# Split data (agar sesuai dengan evaluasi sebelumnya)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Prediksi dan evluasi
y_pred = model.predict(X_test)

# hitung F1 Score 
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

# laporan lengkap 
print("\nClassification Report:")
print(classification_report(y_test, y_pred))