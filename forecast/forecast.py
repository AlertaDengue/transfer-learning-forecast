import sys 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import tensorflow.keras as keras
from datetime import datetime, timedelta

sys.path.append('../')
from lstm import evaluate
from preprocessing import normalize_data 

def get_geocodes_and_state(macro): 
    '''
    This function is used to get the geocodes and state that refer to a specific health macro region code
    
    :param macro:int. A four digit number
        
    '''
    
    dfs = pd.read_csv('../macro_saude.csv')
    
    geocodes = dfs.loc[dfs.code_macro == macro].geocode.unique()
    state = dfs.loc[dfs.code_macro == macro].state.values[0]

    return geocodes, state

def split_data_for(df, look_back=12, ratio=0.8, predict_n=5, Y_column=0):
    """
    Split the data into training and test sets
    Keras expects the input tensor to have a shape of (nb_samples, timesteps, features).
    :param df: Pandas dataframe with the data.
    :param look_back: Number of weeks to look back before predicting
    :param ratio: fraction of total samples to use for training
    :param predict_n: number of weeks to predict
    :param Y_column: Column to predict
    :return:
    """

    s = get_next_n_weeks(ini_date = str(df.index[-1])[:10], next_days= predict_n)

    df = pd.concat([df,pd.DataFrame(index=s)])

    df = np.nan_to_num(df.values).astype("float64")
    # n_ts is the number of training samples also number of training sets
    # since windows have an overlap of n-1
    n_ts = df.shape[0] - look_back - predict_n + 1
    # data = np.empty((n_ts, look_back + predict_n, df.shape[1]))
    data = np.empty((n_ts, look_back + predict_n, df.shape[1]))
    for i in range(n_ts):  # - predict_):
        #         print(i, df[i: look_back+i+predict_n,0])
        data[i, :, :] = df[i: look_back + i + predict_n, :]
    # train_size = int(n_ts * ratio)
    #print(train_size)

    # We are predicting only column 0    
    X_for = data[-1:, :look_back, ]

    #print(X_for.shape)

    #print(X_for[:,:, Y_column])

    return X_for

def get_nn_data_for(city, ini_date = None, end_date = None, look_back = 4, predict_n = 4, filename = None ):
    """
    :param city: int. The ibge code of the city, it's a seven number code 
    :param ini_date: string or None. Initial date to use when creating the train/test arrays 
    :param end_date: string or None. Last date to use when creating the train/test arrays
    :param end_train_date: string or None. Last day used to create the train data 
    :param ratio: float. If end_train_date is None, we use the ratio to spli the data into train and test 
    :param look_back: int. Number of last days used to make the forecast
    :param predict_n: int. Number of days forecast

    """
    df = pd.read_csv(filename, index_col = 'Unnamed: 0' )
    df.index = pd.to_datetime(df.index)  

    try:
        target_col = list(df.columns).index("casos")
    except: 
        target_col = list(df.columns).index(f"casos_{city}")

    #print('Target column:', target_col)

    df = df.dropna()

    if ini_date != None: 
        df = df.loc[ini_date:]

    if end_date != None:
        df = df.loc[:end_date]

        
    norm_df, max_features = normalize_data(df, ratio = 1)
    factor = max_features[target_col]

    X_for = split_data_for(
                norm_df,
                look_back= look_back,
                ratio=1,
                predict_n = predict_n, 
                Y_column=target_col,
        )


    return X_for, factor

def get_next_n_weeks(ini_date: str, next_days: int) -> list:
    """
    Return a list of dates with the {next_days} days after ini_date.
    This function was designed to generate the dates of the forecast
    models.
    Parameters
    ----------
    ini_date : str
        Initial date.
    next_days : int
        Number of days to be included in the list after the date in
        ini_date.
    Returns
    -------
    list
        A list with the dates computed.
    """

    next_dates = []

    a = datetime.strptime(ini_date, "%Y-%m-%d")

    for i in np.arange(1, next_days + 1):
        d_i = datetime.strftime(a + timedelta(days=int(i*7)), "%Y-%m-%d")

        next_dates.append(datetime.strptime(d_i, "%Y-%m-%d").date())

    return next_dates


def apply_forecast(city, ini_date, end_date, look_back, predict_n, filename, model_name):

    #(city, ini_date = None, end_date = None, look_back = 4, predict_n = 4, filename = filename
    X_for, factor = get_nn_data_for(city,
                                                    ini_date = ini_date, end_date = end_date,
                                                    look_back = look_back,
                                                    predict_n = predict_n,
                                                    filename = filename
                                                    )
    #print(X_for.shape)
    model = keras.models.load_model(f'../saved_models/lstm/{model_name}.keras', compile =False)

    pred = evaluate(model, X_for, batch_size = 1)  

    df_pred = pd.DataFrame(np.percentile(pred, 50, axis=2)) * factor
    df_pred2_5 = pd.DataFrame(np.percentile(pred, 2.5, axis=2)) * factor
    df_pred97_5 = pd.DataFrame(np.percentile(pred, 97.5, axis=2)) * factor
    df_pred25 = pd.DataFrame(np.percentile(pred, 25, axis=2)) * factor
    df_pred75 = pd.DataFrame(np.percentile(pred, 75, axis=2)) * factor

    df = create_df_for(end_date, predict_n , df_pred, df_pred2_5, df_pred25, df_pred75, df_pred97_5)

    if len(str(city)) == 4:
        df['macroregion'] = city 

    if len(str(city)) == 7:
        df['city'] = city  
         

    df.to_csv(f'./forecast_tables/forecast_{city}.csv')
    return df

def plot_for(df_data, city, df_pred, df_pred25, df_pred975, ini_date, end_date, target_name, title):
    fig, ax = plt.subplots()

    for_dates = get_next_n_weeks(f'{end_date}', 4)

    ax.plot(df_data[ini_date:][f'{target_name}_{city}'], label = 'Data', color = 'black')


    ax.plot(for_dates, df_pred.iloc[-1].values , color = 'tab:red', label = 'Forecast')

    ax.fill_between(for_dates, df_pred25.iloc[-1].values, 
                    df_pred975.iloc[-1].values, color = 'tab:red', alpha = 0.3)

    ax.legend()

    ax.grid()

    ax.set_title(f'Forecast {title}')

    for tick in ax.get_xticklabels():
            tick.set_rotation(45)

    plt.show()


def plot_for2(df_data, city, df_pred, df_pred25, df_pred975, ini_date, end_date, target_name, title):
    fig, ax = plt.subplots()

    for_dates = get_next_n_weeks(f'{end_date}', 4)

    ax.plot(df_data[ini_date:][f'casos_{city}'], label = 'Data', color = 'black')


    ax.plot(for_dates, df_pred.iloc[-1].values , color = 'tab:red', label = 'Forecast')

    ax.fill_between(for_dates, df_pred25.iloc[-1].values, 
                    df_pred975.iloc[-1].values, color = 'tab:red', alpha = 0.3)

    ax.legend()

    ax.grid()

    ax.set_title(f'Forecast {title}')

    for tick in ax.get_xticklabels():
            tick.set_rotation(45)

    plt.show()


def create_df_for(ini_date_for, predict_n , df_pred, df_pred2_5, df_pred25, df_pred75, df_pred975):

    df = pd.DataFrame()

    for_dates = get_next_n_weeks(f'{ini_date_for}', predict_n)

    df['date'] = for_dates
    df['lower_2_5'] = df_pred2_5.iloc[-1].values
    df['lower_25'] = df_pred25.iloc[-1].values
    df['forecast'] = df_pred.iloc[-1].values
    df['upper_75'] = df_pred75.iloc[-1].values
    df['upper_97_5'] = df_pred975.iloc[-1].values

    return df 


def plot_for_macro(macro, df_for, df_muni, ini_date = '2022-01-01', filename= None, plot = False):

    geocodes, state = get_geocodes_and_state(macro)

    df_data = pd.read_csv(filename, index_col = 'Unnamed: 0')

    df_data.index = pd.to_datetime(df_data.index)

    fig, ax = plt.subplots()

    ax.plot(df_data[ini_date:][f'casos_{macro}'], label = 'Data', color = 'black')

    ax.plot(df_for.date, df_for.forecast , color = 'tab:red', label = 'Forecast')

    ax.fill_between(df_for.date, df_for.lower_2_5, 
                        df_for.upper_97_5, color = 'tab:red', alpha = 0.3)

    ax.fill_between(df_for.date, df_for.lower_25, 
                        df_for.upper_75, color = 'tab:red', alpha = 0.3)

    ax.legend()

    ax.grid()

    ax.set_title(f'Forecast de casos notificados - {macro} - {state}')

    for tick in ax.get_xticklabels():
                tick.set_rotation(20)

    ax.set_ylabel('Novos casos')

    ax.set_xlabel('Data')

    l, b, h, w = .07, .7, .15, .3
    ax2 = fig.add_axes([l, b, w, h])

    df_muni.loc[df_muni.abbrev_state == state].plot(ax = ax2, color = 'lightgray')

    df_muni.loc[df_muni.code_muni.isin(geocodes)].plot(ax = ax2, color = 'red')

    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.savefig(f'./plots/forecast_{macro}_{state}.png', dpi = 300, bbox_inches = 'tight')

    if plot:
        plt.show()


def apply_forecast_macro(macro, ini_date, end_date, look_back, predict_n, filename, model_name, df_muni, plot = False): 
     
    df_for = apply_forecast(macro, ini_date, end_date, look_back=look_back, predict_n=predict_n, filename=filename, model_name = model_name)

    plot_for_macro(int(macro), df_for, df_muni, ini_date = '2022-01-01', filename= filename, plot = plot)

    return df_for