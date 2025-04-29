# import library yang dibutuhkan 
import numpy as np                               # Library untuk komputasi numerik 
import pandas as pd                              # Library untuk pengolahan datasheet 
from sklearn.preprocessing import StandardScaler # Library untuk normalisasi dengan z score 

# load data 
def load_data(path):
    data = pd.read_csv(path)
    return data 

def clean_data(data):
    """Ganti nol pada kolom tertentu jadi NaN, lalu isi dengan median"""
    cols_with_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    for col in cols_with_zero:
        data[col] = data[col].replace(0, np.nan)
        data[col] = data[col].fillna(data[col].median())
    return data 

def remove_outliers(data):
    """Gunakan metode IQR untuk hapus outlier"""
    Q1 = data.quantile(0.25)
    Q3 = data.quantike(0.75)
    IQR = Q3 - Q1
    
    # Hanya ambil data di dalam batas 
    