'''
Train pgbm models on dengue and chik data
'''
 
import pandas as pd
import os
from train_models import train_pgbm_model

PREDICT_N = 4
REFIT=True

df = pd.read_csv('selected_cities.csv', index_col = 'Unnamed: 0')

end_date = '2023-11-01'

for city, ini_date, end_date_train in zip(
    df.geocode, df.start_train_chik, df.end_train_chik): 

    # train dengue model
    if REFIT or not os.path.exists(f'../saved_models/pgbm/{city}_dengue_{PREDICT_N}_pgbm.pt'):
        train_pgbm_model(city, 'dengue', ini_date = None,
                     end_date_train = end_date_train , end_date = end_date)
    
    # train chik model
    if REFIT or not os.path.exists(f'../saved_models/pgbm/{city}_chik_{PREDICT_N}_pgbm.pt'):
        train_pgbm_model(city, 'chik', ini_date = ini_date,
                     end_date_train = end_date_train , end_date = end_date)
    
    

