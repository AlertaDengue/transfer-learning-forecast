import sys
import tensorflow.keras as keras
import pandas as pd

sys.path.append('../')
from plots_lstm import plot_loss
from lstm import build_model, train_model

PREDICT_N = 10 # number of new days predicted
LOOK_BACK = 52  # number of last days used to make the prediction
BATCH_SIZE = 1
EPOCHS = 300
HIDDEN = 16
L1 = 1e-5
L2 = 1e-5


def train_dl_model(city, doenca='dengue', end_date_train='2022-11-01', ratio = None, end_date='2023-11-01', ini_date=None, plot=True, lr =0.0001, filename_data = f'../data/dengue.csv', patience = 40, min_delta = 0.002, label = 'msle',
                      look_back = LOOK_BACK, predict_n = PREDICT_N, hidden = HIDDEN, l1 = L1, l2 = L2, batch_size = BATCH_SIZE, epochs = EPOCHS):
    
    df = pd.read_csv(filename_data, index_col='Unnamed: 0',  nrows = 1)
    
    if len(str(city)) == 4: # is a macroregion 
        
        c = df.columns[df.columns.str.startswith('casos')].shape[0]-2
        
        feat = c*(18) + 2 + 17 + 2   # number of features

    model = build_model(l1=l1, l2=l2, hidden=hidden, features=feat, predict_n=predict_n, look_back=look_back,
                        batch_size=batch_size, loss='msle', lr=lr)

    model, hist = train_model(model, city, doenca=doenca, epochs=epochs, end_train_date=end_date_train,
                                              ini_date=ini_date,
                                              ratio=ratio, end_date=end_date,
                                              predict_n=predict_n, look_back=look_back, label=label,
                                              batch_size=batch_size,
                                              filename=filename_data, verbose=1, patience = patience, min_delta=min_delta)
    
    if plot:
        plot_loss(hist, title=F'Model loss - MSLE - {city}')


