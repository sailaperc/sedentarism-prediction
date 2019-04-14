from matplotlib import pyplot as plt
import pandas as pd
from pandas.plotting import autocorrelation_plot
from seasonality.seasonality_plots import plot_by_week
import numpy as np
def get_user_data(data, userId):
    try:
        return data.loc[data.index.get_level_values(0) == userId].copy()
    except KeyError:
        print('El usuario ', userId, ' no existe.')
lista = []
df = pd.read_pickle('./pkl/dataset.pkl')
for i in df.index.get_level_values(0).drop_duplicates():
    dfu = get_user_data(df, i).droplevel(0).loc[:, 'slevel']
    idx = pd.date_range('2013-03-27 04:00:00', '2013-06-01 3:00:00', freq='h')
    d = pd.DataFrame(index=idx)
    d['slevel'] = dfu
    a = d.isna().sum()
    lista.append(a)

    nulls = pd.DataFrame(index = idx)
    nulls['nulls'] = d.isna()
    nulls['day'] = [t.strftime('%w') for t in nulls.index]
    nulls['week'] = [t.strftime('%U') for t in nulls.index]
    nulls = nulls[nulls['nulls']]
    nulls = nulls.astype(np.float)

    plt.close()
    plt.scatter(nulls['day'], y=nulls['week'],  marker='x')
    plt.grid(True)
    plt.show()

    d.ffill(inplace=True)


    # Autocorrelation Plot
    plt.close()
    plt.rcParams.update({'figure.figsize':(9,5), 'figure.dpi':120})
    autocorrelation_plot(d.slevel)
    plt.title('{0},{1}'.format(str(i),a))
    plt.show()

    plot_by_week(i)



