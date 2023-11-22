import pickle
import numpy as np
from preprocessing import get_ml_data
from plots_pgbm import plot_prediction
# from pgbm.torch import PGBMRegressor,PGBM
from pgbm.sklearn import HistGradientBoostingRegressor


MAIN_FOLDER = '../'

def pgbm_pred(city, state, predict_n, look_back, doenca = 'dengue', ratio = 0.75, ini_date = None, 
                  end_train_date = None, end_date = None, label = 'all', filename = None, verbose = 1):
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

    preds = np.empty((len(X_data), predict_n))
    preds25 = np.empty((len(X_data), predict_n))
    preds975 = np.empty((len(X_data), predict_n))

    for d in range(1, predict_n + 1):
        tgt = targets[d][:len(X_train)]

        # model = HistGradientBoostingRegressor(objective = 'mse', n_estimators= 50,  distribution='poisson', verbose = verbose)
        model = HistGradientBoostingRegressor(distribution='poisson', verbose = verbose)

        model.fit(X_train[:len(tgt)], tgt)
        model
        model.save(f'{MAIN_FOLDER}/saved_models/pgbm/{city}_{doenca}_city_model_{d}_pgbm.pt')

        pred = model.predict(X_data[:len(targets[d])].values)
        
        pred_dist = model.predict_dist(X_data[:len(targets[d])].values)
        
        pred25 = pred_dist.max(axis=0)
        pred = pred
        pred975 = pred_dist.min(axis=0)
        dif = len(X_data) - len(pred)
        if dif > 0:
            pred = list(pred) + ([np.nan] * dif)
            pred25 = list(pred25) + ([np.nan] * dif)
            pred975 = list(pred975) + ([np.nan] * dif)
        preds[:, (d - 1)] = pred
        preds25[:, (d - 1)] = pred25
        preds975[:, (d - 1)] = pred975

    x, y, y25, y975 = plot_prediction(preds, preds25, preds975, target, f'Predictions for {city}', state, len(X_train), doenca,
                                    label = f'pgbm_{city}')

    with open(f'{MAIN_FOLDER}/predictions/pgbm/pgbm_{city}_{doenca}_{label}_predictions.pkl', 'wb') as f:
        pickle.dump({'target':target,'dates': x, 'preds': y, 'preds25': y25,
                    'preds975': y975, 'train_size': len(X_train)
                    }, f)

    return preds, preds25, preds975, X_train, targets 


def cross_dengue_chik_prediction(city, state, predict_n, look_back, ini_date = '2020-01-01', end_date = None, filename = None):
    """
    Functio to apply a model trained with dengue data in chik data. 
    """
    X_data, X_train, targets, target = get_ml_data(city, ini_date = ini_date, end_train_date = None, end_date = end_date, 
                                        ratio = 0.99, predict_n = predict_n, look_back = look_back, filename = filename)

    preds = np.empty((len(X_data), predict_n))
    preds25 = np.empty((len(X_data), predict_n))
    preds975 = np.empty((len(X_data), predict_n))

    for d in range(1, predict_n + 1):
        #print(d)

        model = HistGradientBoostingRegressor(init_model = f'{MAIN_FOLDER}/saved_models/pgbm/{city}_dengue_city_model_{d}_pgbm.pt')
        
        #model.load(f'{MAIN_FOLDER}/saved_models/pgbm/{city}_dengue_city_model_{d}_pgbm.pt')
        
        #pred = model.predict(X_data[:len(targets[d])].values)
        
        #pred_dist = model.predict_dist(X_data[:len(targets[d])].values)
        
        #pred25 = pred_dist.max(axis=0)
        #pred = pred
        #pred975 = pred_dist.min(axis=0)
        #dif = len(X_data) - len(pred)
        #if dif > 0:
        #    pred = list(pred) + ([np.nan] * dif)
        #    pred25 = list(pred25) + ([np.nan] * dif)
        #    pred975 = list(pred975) + ([np.nan] * dif)
        #preds[:, (d - 1)] = pred
        #preds25[:, (d - 1)] = pred25
        #preds975[:, (d - 1)] = pred975

        pred = model.predict(X_data[:len(targets[d])].values)
        
        #print(pred.shape)
        
        pred_dist = model.predict_dist(X_data[:len(targets[d])].values)
        
        pred25 = pred_dist.max(axis=0)
        
        print(pred.shape)
        
        #print(pred25)
        
        #print(pred)
        
        pred = pred
        
        pred975 = pred_dist.min(axis=0)

        dif = len(X_data) - len(pred)
        if dif > 0:
            pred = list(pred) + ([np.nan] * dif)
            pred25 = list(pred25) + ([np.nan] * dif)
            pred975 = list(pred975) + ([np.nan] * dif)
        preds[:, (d - 1)] = pred
        preds25[:, (d - 1)] = pred25
        preds975[:, (d - 1)] = pred975
        
    x, y, y25, y975 = plot_prediction(preds, preds25, preds975, target, 'Predictions for chik at ' + str(city) + ' applying the dengue model', state, None, 'chik',
                                    label = f'pgbm_cross_pred_{city}') 

    with open(f'{MAIN_FOLDER}/predictions/pgbm/pgbm_{city}_chik_cross_predictions.pkl', 'wb') as f:
        pickle.dump({'target': target, 'dates': x, 'preds': y, 'preds25': y25,
                    'preds975': y975, 'train_size': len(X_data)
                    }, f)

    return preds, preds25, preds975, X_data, target
