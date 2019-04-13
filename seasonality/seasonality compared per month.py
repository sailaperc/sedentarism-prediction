import pandas as pd
from seasonality.seasonality_plots import plot_by_month

def get_user_data(data, userId):
    try:
        return data.loc[data.index.get_level_values(0) == userId].copy()
    except KeyError:
        print('El usuario ', userId, ' no existe.')


df = pd.read_pickle('./pkl/dataset.pkl')
for user in df.index.get_level_values(0).drop_duplicates():
    plot_by_month()