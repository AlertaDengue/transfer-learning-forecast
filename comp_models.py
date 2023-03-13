import numpy as np 
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import bootstrap
from sklearn.metrics import mean_squared_error as mse 

def ss(forecast, reference, target):
    
    rmse_f = mse(target, forecast, squared = False )
    
    rmse_r = mse(target, reference, squared = False)
    
    return 1 - (rmse_f/rmse_r)

def conf_ss(forecast, reference, target):
    
    #rmse_f = abs(forecast-target)

    rmse_f = []
    for i in np.arange(1, len(forecast)):

        rmse_f.append(mse(target[:i], forecast[:i], squared = False))

    #rmse_r = abs(reference - target)
    rmse_r = []
    for i in np.arange(1, len(reference)):

        rmse_r.append(mse(target[:i], reference[:i], squared = False))
    
    ss = 1 - (np.array(rmse_f)/np.array(rmse_r))

    #convert array to sequence
    ss = (ss,)

    #calculate 95% bootstrapped confidence interval for median

    bootstrap_ci = bootstrap(ss, np.median, confidence_level=0.95,
                         random_state=1, method='percentile')

    return bootstrap_ci

def conf_all_models(data_nn, data_ml, data_tl, ini_evaluate, end_evaluate): 

    ini_index = data_nn['indice'].index(datetime.strptime(ini_evaluate, '%Y-%m-%d'))
    end_index = data_nn['indice'].index(datetime.strptime(end_evaluate, '%Y-%m-%d'))

    ini_index_ml = data_ml['dates'].index(datetime.strptime(ini_evaluate, '%Y-%m-%d'))
    end_index_ml = data_ml['dates'].index(datetime.strptime(end_evaluate, '%Y-%m-%d'))

    NN_ss_ML = conf_ss( target = data_nn['target'][ini_index - 7: end_index - 7, -1] * data_nn['factor'],
                                forecast =  data_nn['pred'].iloc[ini_index - 7: end_index - 7, -1] * data_nn['factor'],
                                reference =data_ml['preds'][ini_index_ml:end_index_ml] )

    print('---------------------------')

    print('NN compared to ML:')
    print('lower:', NN_ss_ML.confidence_interval[0])
    print('upper:', NN_ss_ML.confidence_interval[1])



    TL_ss_NN = conf_ss( target = data_tl['target'][ini_index - 7: end_index - 7, -1] * data_tl['factor'],
                              forecast =  data_tl['pred'].iloc[ini_index - 7: end_index - 7, -1] * data_tl['factor'],
                              reference =data_nn['pred'].iloc[ini_index - 7: end_index - 7, -1] * data_nn['factor'])

    print('---------------------------')

    print('TL compared to NN:')
    print('lower:', TL_ss_NN.confidence_interval[0])
    print('upper:', TL_ss_NN.confidence_interval[1])


    TL_ss_ML = conf_ss( target = data_tl['target'][ini_index - 7: end_index - 7, -1] * data_tl['factor'],
                                forecast =  data_tl['pred'].iloc[ini_index - 7: end_index - 7, -1] * data_tl['factor'],
                                reference =data_ml['preds'][ini_index_ml:end_index_ml] )

    print('---------------------------')

    print('TL compared to ML:')
    print('lower:', TL_ss_ML.confidence_interval[0])
    print('upper:', TL_ss_ML.confidence_interval[1])

    print('---------------------------')

    return 

def conf_all_models_values(data_nn, data_ml, data_tl, ini_evaluate, end_evaluate): 

    ini_index = data_nn['indice'].index(datetime.strptime(ini_evaluate, '%Y-%m-%d'))
    end_index = data_nn['indice'].index(datetime.strptime(end_evaluate, '%Y-%m-%d'))

    ini_index_ml = data_ml['dates'].index(datetime.strptime(ini_evaluate, '%Y-%m-%d'))
    end_index_ml = data_ml['dates'].index(datetime.strptime(end_evaluate, '%Y-%m-%d'))

    NN_ss_ML = conf_ss( target = data_nn['target'][ini_index - 7: end_index - 7, -1] * data_nn['factor'],
                                forecast =  data_nn['pred'].iloc[ini_index - 7: end_index - 7, -1] * data_nn['factor'],
                                reference = data_ml['preds'][ini_index_ml:end_index_ml] )


    TL_ss_NN = conf_ss( target = data_tl['target'][ini_index - 7: end_index - 7, -1] * data_tl['factor'],
                              forecast =  data_tl['pred'].iloc[ini_index - 7: end_index - 7, -1] * data_tl['factor'],
                              reference =data_nn['pred'].iloc[ini_index - 7: end_index - 7, -1] * data_nn['factor'])

    TL_ss_ML = conf_ss( target = data_tl['target'][ini_index - 7: end_index - 7, -1] * data_tl['factor'],
                                forecast =  data_tl['pred'].iloc[ini_index - 7: end_index - 7, -1] * data_tl['factor'],
                                reference =data_ml['preds'][ini_index_ml:end_index_ml] )

    return NN_ss_ML, TL_ss_NN, TL_ss_ML


def ss_all_models(data_nn, data_ml, data_tl, ini_evaluate, end_evaluate): 
    
    ini_index = data_nn['indice'].index(datetime.strptime(ini_evaluate, '%Y-%m-%d'))
    end_index = data_nn['indice'].index(datetime.strptime(end_evaluate, '%Y-%m-%d'))

    ini_index_ml = data_ml['dates'].index(datetime.strptime(ini_evaluate, '%Y-%m-%d'))
    end_index_ml = data_ml['dates'].index(datetime.strptime(end_evaluate, '%Y-%m-%d'))

    NN_ss_ML = ss( target = data_nn['target'][ini_index - 7: end_index - 7, -1] * data_nn['factor'],
                                forecast =  data_nn['pred'].iloc[ini_index - 7: end_index - 7, -1] * data_nn['factor'],
                                reference =data_ml['preds'][ini_index_ml:end_index_ml] )

    print('---------------------------')

    print('NN compared to ML:')
    print('SS:', NN_ss_ML)


    TL_ss_NN = ss( target = data_tl['target'][ini_index - 7: end_index - 7, -1] * data_tl['factor'],
                              forecast =  data_tl['pred'].iloc[ini_index - 7: end_index - 7, -1] * data_tl['factor'],
                              reference =data_nn['pred'].iloc[ini_index - 7: end_index - 7, -1] * data_nn['factor'])

    print('---------------------------')

    print('TL compared to NN:')
    print('SS:', TL_ss_NN)

    TL_ss_ML = ss( target = data_tl['target'][ini_index - 7: end_index - 7, -1] * data_tl['factor'],
                                forecast =  data_tl['pred'].iloc[ini_index - 7: end_index - 7, -1] * data_tl['factor'],
                                reference =data_ml['preds'][ini_index_ml:end_index_ml] )

    print('---------------------------')

    print('TL compared to ML:')
    print('SS:', TL_ss_ML)

    print('---------------------------')

    return 



def plot_comp(data_nn, data_ml, data_tl, ini_evaluate, end_evaluate): 


    plt.figure()

    ini_index = data_nn['indice'].index(datetime.strptime(ini_evaluate, '%Y-%m-%d'))
    end_index = data_nn['indice'].index(datetime.strptime(end_evaluate, '%Y-%m-%d'))

    plt.plot(data_nn['indice'][ini_index:end_index], data_nn['target'][ini_index - 7: end_index - 7, -1] * data_nn['factor'] , color = 'black', label = 'data', linewidth=2)

    plt.plot(data_nn['indice'][ini_index:end_index],data_nn['pred'].iloc[ini_index - 7: end_index - 7,-1] * data_nn['factor'], label = 'NN', ls = 'dashed', linewidth=2)

    plt.plot(data_tl['indice'][ini_index:end_index], data_tl['pred'].iloc[ini_index - 7: end_index - 7,-1] * data_tl['factor'], label = f'TL', linewidth=2,
                        color = 'tab:red', ls = 'dashdot')

    ini_index_ml = data_ml['dates'].index(datetime.strptime(ini_evaluate, '%Y-%m-%d'))
    end_index_ml = data_ml['dates'].index(datetime.strptime(end_evaluate, '%Y-%m-%d'))

    plt.plot(data_ml['dates'][ini_index_ml:end_index_ml],data_ml['preds'][ini_index_ml: end_index_ml], label = 'ML', ls = 'dotted', linewidth=2)

    plt.grid()
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

    print('RMSE - ML:', mse( data_nn['target'][ini_index - 7: end_index - 7, -1] * data_nn['factor'], data_ml['preds'][ini_index_ml: end_index_ml] , squared = False))
    print('RMSE - NN:', mse( data_nn['target'][ini_index - 7: end_index - 7, -1] * data_nn['factor'], data_nn['pred'].iloc[ini_index - 7: end_index - 7,-1] * data_nn['factor'] , squared = False))
    print('RMSE - TL:', mse( data_nn['target'][ini_index - 7: end_index - 7, -1] * data_nn['factor'], data_tl['pred'].iloc[ini_index - 7: end_index - 7,-1] * data_tl['factor'] , squared = False))