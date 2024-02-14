import pandas as pd
import geopandas as gpd
from forecast import apply_forecast_macro, apply_forecast_state

# from train_models_to_for import LOOK_BACK, PREDICT_N
PREDICT_N = 10  # number of new days predicted
LOOK_BACK = 12

df_muni = gpd.read_file('../muni_br.gpkg')

dfs = pd.read_csv('../macro_saude.csv')
ini_date = None
end_date = '2024-01-28'
# for macro in dfs.loc[dfs.state=='MG'].code_macro.unique():
for macro in dfs.code_macro.unique():
    print(f'Forecasting: {macro}')

    filename = f'../data/dengue_{macro}.csv.gz'
    model_name = f'trained_{macro}_dengue_macro'

    df_for = apply_forecast_macro(macro, ini_date, end_date, look_back=LOOK_BACK, predict_n=PREDICT_N,
                                  filename=filename, model_name=model_name, df_muni=df_muni)
        


for state in dfs.state.unique():
    print(f'Forecasting: {state}')

    filename = f'../data/dengue_{state}.csv.gz'
    model_name = f'trained_{state}_dengue_state'

    df_for = apply_forecast_state(state, ini_date, end_date, look_back=LOOK_BACK, predict_n=PREDICT_N,
                                  filename=filename, model_name=model_name, gen_fig = True)
        
