import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import epiweeks
import geopandas as gpd
import altair as alt

code_to_state = {33: 'RJ', 32: 'ES', 41: 'PR', 23: 'CE', 21: 'MA',
 31: 'MG', 42: 'SC', 26: 'PE', 25: 'PB', 24: 'RN', 22: 'PI', 27: 'AL',
 28: 'SE', 35: 'SP', 43: 'RS', 15: 'PA', 16: 'AP', 14: 'RR',  11: 'RO',
 13: 'AM', 12: 'AC', 51: 'MT', 50: 'MS', 52: 'GO', 17: 'TO', 53: 'DF',
 29: 'BA'}

for i, m in enumerate(glob.glob('./forecast_tables/forecast_*.csv.gz')):
    if i == 0:
        df = pd.read_csv(m)
        dates = df.date.unique()
    else:
        df = pd.concat([df, pd.read_csv(m)])

df.prob_low = -df.prob_low
df['prob_color'] = df.apply(lambda x: x.prob_low if abs(x.prob_low) > abs(x.prob_high) else x.prob_high, axis=1)
df['prob_color'] = df.prob_color.apply(lambda x: 0 if abs(x) < 50 else x)

df_macro = gpd.read_file('./shapefile_macro.gpkg')

df_macro = df_macro.dropna()

df_macro['state'] = (df_macro.code_macro.astype(str).str[:2]).astype(int).replace(code_to_state)

 # [df.date.isin(df.date.unique()[:2])]
df_macro = df_macro.merge(df, left_on='code_macro', right_on='macroregion', how='left')


df_macro['desc_prob'] = np.nan

df_macro.loc[df_macro.prob_color > 0, 'desc_prob'] =   'Probabilidade de a incidência superar o limiar histórico'

df_macro.loc[df_macro.prob_color < 0, 'desc_prob'] =   'Probabilidade de a incidência ser abaixo do limiar inferior histórico'

select_date = alt.selection_point(
    name="date",
    fields=["date"],
    bind = alt.binding_select(options=df_macro.date.unique(), name='date'),
    value='2024-01-28',
)

map_dist = alt.Chart(df_macro, title='').mark_geoshape().encode(
    color= alt.Color('HTinc:Q',
              scale=alt.Scale(scheme='Viridis'),
             legend=alt.Legend(direction='vertical', orient='right', legendY=30, 
        title = 'Incidência /100.000 hab.', titleOrient = 'left')), 
    
    tooltip = [alt.Tooltip('state:N', title='Estado:'),
               alt.Tooltip('name_code_macro:N', title='Macrorregião:'),
               alt.Tooltip('HTinc:Q', title='Incidência:')]).add_params(select_date).transform_filter(select_date)


text_dist = alt.Chart(df_macro).mark_text(dy=-170, dx = 20, size=14, fontWeight=100).encode(
    text='date:N'
).transform_filter(
    select_date
).transform_calculate(date = '"Limiar superior de Incidência na semana de " + datum.date')

#states_line = alt.Chart(df_states).mark_geoshape(
 #   filled=False,
  #  strokeWidth=1.5,
   # color = 'black'
#)

map_prob = alt.Chart(df_macro, title='').mark_geoshape().encode(
    color= alt.Color('prob_color:Q',
              scale=alt.Scale(scheme='redblue',reverse=True),
                    legend=alt.Legend(direction='vertical', orient='right', legendY=30, 
        title = 'Probabilidade (%)', titleOrient = 'left')),
    tooltip = [alt.Tooltip('state:N', title='Estado:'),
                alt.Tooltip('name_code_macro:N', title='Macrorregião:'),
               alt.Tooltip('prob_color:Q', title='Probabilidade (%):'),
              alt.Tooltip('desc_prob:N', title = 'Info:')]
).add_params(select_date).transform_filter(select_date)


text_prob = alt.Chart(df_macro).mark_text(dy=-170, dx = 20, size=14, fontWeight=100).encode(
    text='date:N'
).transform_filter(
    select_date
).transform_calculate(date = '"Previsão probabilística na semana de " + datum.date')


final_maps = alt.hconcat(alt.layer(map_dist, text_dist), alt.layer(map_prob,text_prob)).resolve_scale(color='independent')


# Base chart for data tables
ranked_table_prob = alt.Chart(df_macro[['date', 'state', 'name_code_macro', 'prob_color', 'HTinc']]).mark_text(align='right').encode(
    y=alt.Y('row_number:O').axis(None)
).add_params(select_date).transform_filter(
    select_date
).transform_window(
    row_number='row_number()',
    rank='rank(prob_color)',
    sort=[alt.SortField('prob_color', order='descending')]
    
).transform_filter(
    alt.datum.prob_color > 90
)


# Data Tables
d = ranked_table_prob.encode(text='date:N').properties(
    title=alt.Title(text='date', align='right')
)
name = ranked_table_prob.encode(text='name_code_macro:N').properties(
    title=alt.Title(text='Macrorregião', align='right')
)

state = ranked_table_prob.encode(text='state:N').properties(
    title=alt.Title(text='Estado', align='right')
)

prob = ranked_table_prob.encode(text='prob_color:N').properties(
    title=alt.Title(text='Probabilidade (%)', align='right')
)

inc =  ranked_table_prob.encode(text='HTinc:Q').properties(
    title=alt.Title(text='Limiar superior de Incidência (100k)', align='right')
)

table_prob = alt.hconcat(d, state, name, inc, prob) # Combine data tables

#table_prob = table_prob.configure_view(
 #   stroke=None
#)


final_plot = alt.vconcat(final_maps, table_prob).configure_view(
   stroke=None)

final_plot.save('map_macro.html',  embed_options={'renderer':'svg'})