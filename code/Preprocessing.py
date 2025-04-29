# import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Load data
def load_data(path):
    """
    Load data dari CSV
    Args:
        path (str): path ke file CSV
    Returns:
        pd.DataFrame: dataset
    """
    data = pd.read_csv(path)
    return data

# 2. Clean data
def clean_data(data):
    """
    Replace 0 with median in specific columns
    Args:
        data (pd.DataFrame): original data
    Returns:
        pd.DataFrame: cleaned data
    """
    cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zero:
        data[col] = data[col].replace(0, np.nan)
        data[col] = data[col].fillna(data[col].median())
    return data

# 3. Remove outliers
def remove_outliers(data):
    """
    Remove outliers using IQR method
    Args:
        data (pd.DataFrame): cleaned data
    Returns:
        pd.DataFrame: data without outliers
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    condition = ~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)
    data = data[condition]
    return data

# 4. Normalize features
def normalize_data(X):
    """
    Normalize features with StandardScaler
    Args:
        X (np.ndarray or pd.DataFrame): feature matrix
    Returns:
        tuple: (X_scaled, fitted scaler)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# Example penggunaan
if __name__ == "__main__":
    data = load_data('diabetes.csv')
    data = clean_data(data)
    data = remove_outliers(data)
    
    X = data.drop('Outcome', axis=1)  # fitur
    y = data['Outcome']               # label
    
    X_scaled, scaler = normalize_data(X)
