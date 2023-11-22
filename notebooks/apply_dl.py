'''
Apply the DL models trained on dengue and chik data
'''
 

import sys 
import pandas as pd
from train_models import LOOK_BACK, PREDICT_N

sys.path.append('../')
from lstm import apply_model


df = pd.read_csv('selected_cities.csv', index_col = 'Unnamed: 0')

end_date = '2023-11-01'

for city, ini_date, end_date_train in zip(
    df.geocode, df.start_train_chik, df.end_train_chik): 

    # apply dl on dengue data 

    metrics = apply_model(city, ini_date = None, 
                    end_date = end_date, look_back = LOOK_BACK, end_train_date =  end_date_train, batch_size = 1, 
                    predict_n = PREDICT_N,  ratio = None,
                    label_pred= 'dengue_pred',
                    model_name = f'trained_{city}_dengue_msle', 
                    filename = f'../data/dengue_{city}_cluster.csv', plot = False)
    

    # apply dl dengue on chik data 
    metrics = apply_model(city, ini_date = ini_date, 
                    end_date = end_date, look_back = LOOK_BACK, end_train_date =  end_date_train, batch_size = 1, 
                    predict_n = PREDICT_N,  ratio = None,
                    label_pred= 'chik_dengue_pred',
                    model_name = f'trained_{city}_dengue_msle', 
                    filename = f'../data/chik_{city}_cluster.csv', plot = False)
    

    # apply transfer 

    metrics = apply_model(city, ini_date = ini_date, 
                    end_date = end_date, look_back = LOOK_BACK, end_train_date =  end_date_train, batch_size = 1, 
                    predict_n = PREDICT_N,  ratio = None,
                    label_pred= 'chik_transf_pred',
                    model_name = f'trained_{city}_chik_transf_msle', 
                    filename = f'../data/chik_{city}_cluster.csv', plot = False)
    

    # apply dl chik on chik 

    metrics = apply_model(city, ini_date = ini_date, 
                    end_date = end_date, look_back = LOOK_BACK, end_train_date =  end_date_train, batch_size = 1, 
                    predict_n = PREDICT_N,  ratio = None,
                    label_pred= 'chik_pred',
                    model_name = f'trained_{city}_chik_msle', 
                    filename = f'../data/chik_{city}_cluster.csv', plot = False)
    
