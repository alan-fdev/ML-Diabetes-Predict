import pandas as pd 
import joblib
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression

from Preprocessing import load_data, clean_data, remove_outliers, normalize_data
 
# 1. Load dan preprocess data
data = load_data('../Datasheet/diabetes.csv')
data = clean_data(data)
data = remove_outliers(data)

# 2. Pisahkan fitur dan target 
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# 3. Normalisasi fitur 
X_scaled, scaler = normalize_data(X)

# 4. Split data 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) 

# 5. Setup model & hyperparameter grid 
logreg = LogisticRegression(solver='liblinear', max_iter=500)

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']  
}

grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)
