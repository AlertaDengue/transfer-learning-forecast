import pickle
import joblib
import numpy as np
from preprocessing import get_ml_data
from plots_pgbm import plot_prediction
from pgbm.sklearn import HistGradientBoostingRegressor
from pgbm.sklearn import HistGradientBoostingRegressor


MAIN_FOLDER = '../'

def pgbm_train(city, predict_n, look_back, doenca = 'dengue', ratio = 0.75, ini_date = None, 
                  end_train_date = None, end_date = None, filename = None, verbose = 0):
    """
    Train a model for a single city and disease.
    :param city:
    :param state:
    :param predict_n:
    :param look_back:
    :return:
    """
 
    X_data, X_train, targets, target = get_ml_data(city, ini_date = ini_date, end_train_date = end_train_date, end_date = end_date, 
                                        ratio = ratio , predict_n = predict_n, look_back = look_back, filename = filename)


    for d in range(1, predict_n + 1):
        tgt = targets[d][:len(X_train)]

        model = HistGradientBoostingRegressor(random_state=0, l2_regularization = 0.1, distribution='negativebinomial', verbose = verbose)
        
        model.fit(X_train[:len(tgt)], tgt)

        joblib.dump(model, f'{MAIN_FOLDER}/saved_models/pgbm/{city}_{doenca}_{d}_pgbm.pt')

    return model


def pgbm_pred(city, predict_n, look_back, doenca = 'dengue', ratio = 0.75, ini_date = None, 
                  end_train_date = None, end_date = None, filename = None, plot = True):
    
    
    X_data, X_train, targets, target = get_ml_data(city, ini_date = ini_date, end_train_date = end_train_date, end_date = end_date, 
                                        ratio = ratio , predict_n = predict_n, look_back = look_back, filename = filename)

    preds = np.empty((len(X_data), predict_n))
    preds25 = np.empty((len(X_data), predict_n))
    preds975 = np.empty((len(X_data), predict_n))

    for d in range(1, predict_n + 1):
        tgt = targets[d][:len(X_train)]

        model = joblib.load( f'{MAIN_FOLDER}/saved_models/pgbm/{city}_{doenca}_{d}_pgbm.pt')
        model = HistGradientBoostingRegressor(distribution='poisson', verbose = verbose)

        pred, pred_std = model.predict(X_data[:len(targets[d])].values, return_std=True)
        
        pred[pred < 0 ] = 0.1
        ensemble = model.sample(pred, pred_std, n_estimates=10_000, random_state=0)


        if d == 4:
            ensemble_to_save = ensemble

        pred25 = np.percentile(ensemble, 2.5, axis=0)
        pred = np.percentile(ensemble, 50, axis=0)
        pred975 = np.percentile(ensemble, 97.5, axis=0)
       
    
        dif = len(X_data) - len(pred)
        if dif > 0:
            pred = list(pred) + ([np.nan] * dif)
            pred25 = list(pred25) + ([np.nan] * dif)
            pred975 = list(pred975) + ([np.nan] * dif)
        preds[:, (d - 1)] = pred
        preds25[:, (d - 1)] = pred25
        preds975[:, (d - 1)] = pred975
        

    if plot:
        state = ''
        x, y, y25, y975 = plot_prediction(preds, preds25, preds975, target, f'Predictions for {city}', state, len(X_train), doenca,
                                            label = f'pgbm_{city}')
    
    else: 
        # configure predictions 
        ydata = target
        pred_window = preds.shape[1]
        llist = range(len(ydata.index) - (preds.shape[1]))
        x = []
        y = []
        y25 = []
        y975 = []
        for n in llist:
            
            x.append(ydata.index[n + pred_window])
            y.append(preds[n][-1])
            y25.append(preds25[n][-1])
            y975.append(preds975[n][-1])

    with open(f'{MAIN_FOLDER}/predictions/pgbm/pgbm_{city}_{doenca}_pred.pkl', 'wb') as f:
        pickle.dump({'target':target,'dates': x, 'preds': y, 'preds25': y25,
                    'preds975': y975, 'train_size': len(X_train), 'ensemble': ensemble_to_save
                    }, f)
        

    return #model 

        

def cross_dengue_chik_prediction(city, predict_n, look_back, ini_date = '2020-01-01', end_date = None, filename = None, plot = True):
    """
    Function to apply a model trained with dengue data in chik data. 
    """
    X_data, X_train, targets, target = get_ml_data(city, ini_date = ini_date, end_train_date = None, end_date = end_date, 
                                        ratio = 0.99, predict_n = predict_n, look_back  = look_back, filename = filename)

    preds = np.empty((len(X_data), predict_n))
    preds25 = np.empty((len(X_data), predict_n))
    preds975 = np.empty((len(X_data), predict_n))

    for d in range(1, predict_n + 1):
        tgt = targets[d][:len(X_train)]

        model = joblib.load( f'{MAIN_FOLDER}/saved_models/pgbm/{city}_dengue_{d}_pgbm.pt')

        pred, pred_std = model.predict(X_data[:len(targets[d])].values, return_std=True)
        
        pred[pred < 0 ] = 0.1
        ensemble = model.sample(pred, pred_std, n_estimates=10_000, random_state=0)


        if d == 4:
            ensemble_to_save = ensemble

        pred25 = np.percentile(ensemble, 2.5, axis=0)
        pred = np.percentile(ensemble, 50, axis=0)
        pred975 = np.percentile(ensemble, 97.5, axis=0)
       
    
        dif = len(X_data) - len(pred)
        if dif > 0:
            pred = list(pred) + ([np.nan] * dif)
            pred25 = list(pred25) + ([np.nan] * dif)
            pred975 = list(pred975) + ([np.nan] * dif)
        preds[:, (d - 1)] = pred
        preds25[:, (d - 1)] = pred25
        preds975[:, (d - 1)] = pred975
        

    if plot:
        state = ''
        x, y, y25, y975 = plot_prediction(preds, preds25, preds975, target, f'Predictions for {city}', state, len(X_train), 'chik',
                                            label = f'pgbm_{city}')
    
    else: 
        # configure predictions 
        ydata = target
        pred_window = preds.shape[1]
        llist = range(len(ydata.index) - (preds.shape[1]))
        x = []
        y = []
        y25 = []
        y975 = []
        for n in llist:
            
            x.append(ydata.index[n + pred_window])
            y.append(preds[n][-1])
            y25.append(preds25[n][-1])
            y975.append(preds975[n][-1])

    with open(f'{MAIN_FOLDER}/predictions/pgbm/pgbm_{city}_chik_cross_pred.pkl', 'wb') as f:
        pickle.dump({'target':target,'dates': x, 'preds': y, 'preds25': y25,
                    'preds975': y975, 'train_size': len(X_train), 'ensemble': ensemble_to_save
                    }, f)

    return preds, preds25, preds975, X_data, target
