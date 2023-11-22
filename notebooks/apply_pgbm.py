'''
Apply pgbm models trained on dengue and chik data
'''
 

import sys 
import pandas as pd
from train_models import LOOK_BACK, PREDICT_N
from train_models import  apply_dengue_pgbm_on_chik
sys.path.append('../')
from pgbm_model import pgbm_pred


df = pd.read_csv('selected_cities.csv', index_col = 'Unnamed: 0')

end_date = '2023-11-01'

for city, ini_date, end_date_train in zip(
    df.geocode, df.start_train_chik, df.end_train_chik): 

    # apply on dengue 
    pgbm_pred(city, PREDICT_N, LOOK_BACK, doenca = 'dengue', ratio = 0.75, ini_date = None, 
                  end_train_date = end_date_train, end_date = end_date,
                  filename =  f'../data/dengue_{city}_cluster.csv', plot = False)
    
    # apply the dengue model on chik
    apply_dengue_pgbm_on_chik(city, ini_date= ini_date, end_date = end_date, plot = False)

    # apply the chik model on chik
    pgbm_pred(city, PREDICT_N, LOOK_BACK, doenca = 'chik', ratio = 0.75, ini_date = ini_date, 
                  end_train_date = end_date_train, end_date = end_date,
                  filename =  f'../data/chik_{city}_cluster.csv', plot = False)
    