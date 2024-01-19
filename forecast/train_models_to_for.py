import pandas as pd 
from train_models import train_dl_model

dfs = pd.read_csv('../macro_saude.csv')


PREDICT_N = 10 # number of new days predicted
LOOK_BACK = 52  # number of last days used to make the prediction
BATCH_SIZE = 1
EPOCHS = 10
HIDDEN = 16
L1 = 1e-5
L2 = 1e-5

for macro in dfs.loc[dfs.state=='MG'].code_macro.unique():

    print(f'Training Macroregion: {macro}')

    FILENAME_DATA = f'../data/dengue_{macro}.csv' 

    end_date = '2023-12-24'

    train_dl_model(macro, doenca = 'dengue',
                  end_date_train = None,
                  ratio = 1, 
                  end_date = end_date,
                  plot = False, filename_data = FILENAME_DATA, 
                  min_delta = 0.001, label = 'macro', 
                  lr = 0.0001,
                  epochs = EPOCHS,
                  hidden = HIDDEN, 
                  l1 = L1, 
                  l2 = L2, 
                  batch_size =BATCH_SIZE,
                 predict_n = PREDICT_N,
                 look_back=LOOK_BACK)
    
