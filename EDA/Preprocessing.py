# import library yang dibutuhkan 
import numpy as np                               # Library untuk komputasi numerik 
import pandas as pd                              # Library untuk pengolahan datasheet 
from sklearn.preprocessing import StandardScaler # Library untuk normalisasi dengan z score 

# load data 
def load_data(path):
    data = pd.read_csv(path)
    return data 
