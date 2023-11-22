'''
Train pgbm models on dengue and chik data
'''
 
import pandas as pd
from train_models import train_pgbm_model

df = pd.read_csv('selected_cities.csv', index_col = 'Unnamed: 0')

end_date = '2023-11-01'

for city, ini_date, end_date_train in zip(
    df.geocode, df.start_train_chik, df.end_train_chik): 

    # train dengue model 
    train_pgbm_model(city, 'dengue', ini_date = None, 
                     end_date_train = end_date_train , end_date = end_date)
    
    # train chik model 
    train_pgbm_model(city, 'chik', ini_date = ini_date, 
                     end_date_train = end_date_train , end_date = end_date)
    
    

