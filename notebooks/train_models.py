import sys
import pandas as pd
sys.path.append('../')
from plots_lstm import plot_loss
from lstm import apply_dengue_chik
from lstm import build_model, make_pred
from lstm import build_model, transf_chik_pred, transf_model
from pgbm_model import pgbm_pred, cross_dengue_chik_prediction

PREDICT_N = 4# number of new days predicted
LOOK_BACK = 52 # number of last days used to make the prediction 
BATCH_SIZE = 1
EPOCHS = 250
HIDDEN =64
L1 = 1e-5
L2 = 1e-5

def train_dengue_model(city,  end_date_train = '2022-11-01', end_date = '2023-11-01', ini_date = None): 
    
    FILENAME_DATA = f'../data/dengue_{city}_cluster.csv'
    cols = pd.read_csv(FILENAME_DATA, index_col = 'Unnamed: 0').shape[1]
    
    FEAT = int((1 +1/16)*cols) +2 # number of features
    
    model = build_model(l1=L1, l2 = L2, hidden = HIDDEN, features = FEAT, predict_n = PREDICT_N, look_back=LOOK_BACK, batch_size=BATCH_SIZE, loss = 'msle', lr = 0.0001)

    m_msle_all, h_msle_all, m_train_all, m_val_all = make_pred(model, city, doenca = 'dengue', epochs = EPOCHS, end_train_date = end_date_train, 
                        ini_date = ini_date,
                                                               ratio= None, end_date = end_date,
                         predict_n = PREDICT_N, look_back =  LOOK_BACK, label = 'msle',  filename = FILENAME_DATA, verbose = 0)

    plot_loss(h_msle_all, title = F'Model loss - MSLE - {city}')
  

def train_chik_model(city, end_date_train = '2022-11-01', end_date = '2023-11-01', ini_date = None): 
    
    FILENAME_DATA = f'../data/chik_{city}_cluster.csv'
    
    cols = pd.read_csv(FILENAME_DATA, index_col = 'Unnamed: 0').shape[1]
    
    FEAT = int((1 +1/16)*cols) +2 # number of features

    model = build_model(l1=L1, l2 = L2, hidden = HIDDEN, features = FEAT, predict_n = PREDICT_N, look_back=LOOK_BACK, batch_size=BATCH_SIZE, loss = 'msle', lr = 0.0001)

    m_msle_all, h_msle_all, m_train_all, m_val_all = make_pred(model, city, doenca = 'chik', epochs = EPOCHS, end_train_date = end_date_train, 
                        ini_date = ini_date,
                                                               ratio= None, end_date = end_date,
                         predict_n = PREDICT_N, look_back =  LOOK_BACK, label = 'msle',  filename = FILENAME_DATA, verbose = 1)

    plot_loss(h_msle_all, title = F'Model loss - MSLE - {city}')


def transf_and_pred_chik(city, ini_date, end_date_train, end_date): 

    FILENAME_DATA = f'../data/chik_{city}_cluster.csv'

    BATCH_SIZE = 1
    
    cols = pd.read_csv(FILENAME_DATA, index_col = 'Unnamed: 0').shape[1]
    
    FEAT = int((1 +1/16)*cols) +2 # number of features
    
    filename = f'../saved_models/lstm/trained_{city}_dengue_msle.h5'
    model = transf_model(filename, L1,L2,HIDDEN, FEAT, PREDICT_N, LOOK_BACK, batch_size = BATCH_SIZE, lr = 0.0001)
    
    # apply transf model 
    m_msle, hist, m_t, m_val = transf_chik_pred(model, city, ini_date = ini_date, end_train_date = end_date_train,  
                                    end_date = end_date,  epochs= EPOCHS, 
                                    predict_n = PREDICT_N, look_back = LOOK_BACK, validation_split = 0.0, monitor = 'loss',
                                      patience = 10, min_delta = 0.001,
                                    label = f'transf_msle', ratio = None, filename_data = FILENAME_DATA,  verbose=1)


    plot_loss(hist, title = f'Model loss - MSLE - {city}')

    
def apply_dl_dengue_on_chik(city, ini_date, end_date_train, end_date): 

    FILENAME_DATA = f'../data/chik_{city}_cluster.csv'

    # apply NN model 
    metrics = apply_dengue_chik(city, ini_date = ini_date, 
                            end_date = end_date, look_back = LOOK_BACK, end_train_date = end_date_train, batch_size = 1, 
                            predict_n = PREDICT_N,  ratio = None, label_m = f'msle', filename = FILENAME_DATA )
    

def train_pgbm_dengue(city, state, ini_date, end_date_train, end_date):
   FILENAME_DATA = f'../data/dengue_{city}_cluster.csv'

   preds, preds25, preds975, X_train, targets = pgbm_pred(city, state, PREDICT_N, LOOK_BACK, doenca = 'dengue', ini_date = ini_date, end_train_date = end_date_train, end_date = end_date,  filename = FILENAME_DATA, verbose = 1)

def apply_dengue_pgbm_on_chik(city, state, ini_date, end_date):
    FILENAME_DATA = f'../data/chik_{city}_cluster.csv'
    preds, preds25, preds975, X_data, targets = cross_dengue_chik_prediction(city, state, predict_n = PREDICT_N, look_back = LOOK_BACK, ini_date = ini_date, end_date = end_date, filename = FILENAME_DATA )
    

def train_pgbm_chik(city, state, ini_date, end_date_train, end_date):
   FILENAME_DATA = f'../data/chik_{city}_cluster.csv'

   preds, preds25, preds975, X_train, targets = pgbm_pred(city, state, PREDICT_N, LOOK_BACK, doenca = 'chik', ini_date = ini_date, end_train_date = end_date_train, end_date = end_date,  filename = FILENAME_DATA)
  
