'''
Train DL models on dengue and chik data an apply the transfer learning
'''
 
import pandas as pd
from train_models import train_dl_model, train_transf_chik


df = pd.read_csv('selected_cities.csv', index_col = 'Unnamed: 0')

end_date = '2023-11-01'

for city, ini_date, end_date_train in zip(
    df.geocode, df.start_train_chik, df.end_train_chik): 

    # train the dengue model  
    train_dl_model(city,   doenca = 'dengue', end_date_train = end_date_train , end_date = end_date, plot = False)

    # apply transfer
    train_transf_chik(city, ini_date = ini_date, end_date_train = end_date_train , end_date = end_date, plot =False)

    # train the chik model 
    train_dl_model(city,   doenca = 'chik', end_date_train = end_date_train , end_date = end_date, plot = False)
