

import numpy as np
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt  

MAIN_FOLDER = '../..'


def plot_prediction(preds, preds25, preds975, ydata, title, state, train_size, doenca, label,  path='quantile_lgbm', save=True):
    plt.figure()
    plt.plot(ydata[4:], 'k-', label='data')


    min_val = min([min(ydata), np.nanmin(preds)])
    max_val = max([max(ydata), np.nanmax(preds)])

    if train_size != None:
        if train_size == len(ydata): 
            point = ydata.index[train_size-1]
            plt.vlines(point, min_val, max_val, 'g', 'dashdot', lw=2, label = 'Train/Test')
        else: 
            point = ydata.index[train_size]
            plt.vlines(point, min_val, max_val, 'g', 'dashdot', lw=2, label = 'Train/Test')

    pred_window = preds.shape[1]
    llist = range(len(ydata.index) - (preds.shape[1]))
    # print(type(preds))

    # # for figure with all predicted points
    # for n in llist:
    #     plt.vlines(ydata.index[n + pred_window], 0, preds[n][-1], 'b', alpha=0.2)
    #     plt.plot(ydata.index[n:n + pred_window], preds[n], 'r')

    # for figure with only the last prediction point (single red line)
    x = []
    y = []
    y25 = []
    y975 = []
    for n in llist:
        # plt.vlines(ydata.index[n + pred_window], 0, preds[n][-1], 'b', alpha=0.2)
        x.append(ydata.index[n + pred_window])

        y.append(preds[n][-1])
        y25.append(preds25[n][-1])
        y975.append(preds975[n][-1])
    plt.plot(x, y, 'r-', alpha=0.5, label='median')
    # plt.plot(x, y25, 'b-', alpha=0.3)
    # plt.plot(x, y975, 'b-', alpha=0.3)
    plt.fill_between(x, np.array(y25), np.array(y975), color='b', alpha=0.3)


    
    plt.grid()
    plt.ylabel('incidence')
    plt.xlabel('time')
    #plt.title(title)
    plt.xticks(rotation=30)
    plt.legend(loc=0)
    if save:
        plt.savefig(f'{MAIN_FOLDER}/plots/qlgbm_qlgbm_{doenca}_{label}_ss.png',bbox_inches='tight', dpi=300)
    plt.show()
    return x, y, y25, y975



def plot_transf_prediction(pred_window, preds_t, preds, ydata, title, state, train_size, doenca,  path='quantile_lgbm', save=True):
    plt.clf()
    plt.plot(ydata, 'k-', label='data')

    point = ydata.index[train_size]

    min_val = min([min(ydata), np.nanmin(preds), np.nanmin(preds_t)])
    max_val = max([max(ydata), np.nanmax(preds), np.nanmax(preds_t)])
    plt.vlines(point, min_val, max_val, 'g', 'dashdot', lw=2, label = 'Train/Test')

    llist = range(len(ydata.index) - (pred_window))
    # print(type(preds))

    # # for figure with all predicted points
    # for n in llist:
    #     plt.vlines(ydata.index[n + pred_window], 0, preds[n][-1], 'b', alpha=0.2)
    #     plt.plot(ydata.index[n:n + pred_window], preds[n], 'r')

    # for figure with only the last prediction point (single red line)
    x = []
    yt = []
    y = []

    for n in llist:
        # plt.vlines(ydata.index[n + pred_window], 0, preds[n][-1], 'b', alpha=0.2)
        x.append(ydata.index[n + pred_window])

        #yt.append(preds_t[n][-1])
        #y.append(preds[n][-1])
    plt.plot(x, preds_t, 'g-', alpha=0.5, label='median - transf model')
    # plt.plot(x, y25, 'b-', alpha=0.3)
    # plt.plot(x, y975, 'b-', alpha=0.3)
    plt.plot(x, preds, 'r-', alpha=0.5, label='median - chik model')

    #plt.text(point, 0.6 * max_val, "Out of sample Predictions")
    plt.grid()
    plt.ylabel('Weekly cases')
    #plt.title('Predictions for {}'.format(title))
    plt.xticks(rotation=0)
    plt.legend(loc=0)
    if save:
        plt.savefig(f'{MAIN_FOLDER}/plots/qlgbm/qlgbm_{doenca}_{title}_ss.png', dpi=300)
    plt.show()
    return None


    

def predicted_vs_observed(predicted, real, city, state, doenca, model_name, city_name,  plot=True):
    """
    Plot QQPlot for prediction values
    :param plot: generates an saves the qqplot when True (default)
    :param predicted: Predicted matrix
    :param real: Array of target_col values used in the prediction
    :param city: Geocode of the target city predicted
    :param state: State containing the city
    :param look_back: Look-back time window length used by the model
    :param all_predict_n: If True, plot the qqplot for every week predicted
    :return:
    """
    # Name = get_city_names([city])
    # data = get_alerta_table(city, state, doenca=doenca)

    obs_preds = np.hstack((predicted, real))
    q_p = [ss.percentileofscore(obs_preds, x) for x in predicted]
    q_o = [ss.percentileofscore(obs_preds, x) for x in real]
    plot_cross_qq(city, doenca, q_o, q_p, model_name, city_name)
    return np.array(q_o), np.array(q_p)


def plot_cross_qq(city, doenca, q_o, q_p,model_name, city_name):
    ax = sns.kdeplot(q_o[len(q_p) - len(q_o):], q_p, shade=True)
    ax.set_xlabel('observed')
    ax.set_ylabel('predicted')
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    plt.plot([0, 100], [0, 100], 'k')
    #plt.title(f'Predictions percentiles with {model_name.lower()} for {doenca} at {city_name}')
    plt.savefig(f'{MAIN_FOLDER}/plots/qlgbm/qlgbm_cross_qqbplot_{model_name}_{doenca}_{city}.png', dpi=300)
