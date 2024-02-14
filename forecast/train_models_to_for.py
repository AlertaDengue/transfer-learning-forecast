import pandas as pd
import os
from train_models import train_dl_model

dfs = pd.read_csv('../macro_saude.csv')

PREDICT_N = 10  # number of new days predicted
LOOK_BACK = 12  # number of last days used to make the prediction
BATCH_SIZE = 1
EPOCHS = 300
#HIDDEN = 32
L1 = 1e-5
L2 = 1e-5
TRAIN_FROM = '2015-06-01'

# for macro in dfs.loc[dfs.state=='MG'].code_macro.unique():
#for macro in dfs.code_macro.unique():
for macro in dfs.state.unique():

    if macro in ['AC', 'AL', 'AP', 'RN', 'RO', 'RR', 'SE', 'TO']:
        HIDDEN = 32
    else:
        HIDDEN = 64 

    print(f'Training Macroregion: {macro}')

    FILENAME_DATA = f'../data/dengue_{macro}.csv.gz'

    end_date = '2023-12-31'
    # if os.path.exists(f'../saved_models/lstm/trained_{macro}_dengue_macro.keras'):
    #     continue

    train_dl_model(macro, doenca='dengue',
                   end_date_train=None,
                   ratio=1,
                   ini_date=TRAIN_FROM,
                   end_date=end_date,
                   plot=False, filename_data=FILENAME_DATA,
                   min_delta=0.01, label='state',
                   lr=0.0001,
                   epochs=EPOCHS,
                   hidden=HIDDEN,
                   l1=L1,
                   l2=L2,
                   batch_size=BATCH_SIZE,
                   predict_n=PREDICT_N,
                   look_back=LOOK_BACK)
