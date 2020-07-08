
#%%
import matplotlib.dates as mdates
from preprocessing.datasets import get_dataset, get_user_data
from utils.utils_graphic import plot_met_statistics,plot_user_activity, plot_user_activity_and_met
from preprocessing.various import get_activity_levels, addSedentaryLevel
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_style("whitegrid")
import locale
locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
#%%
df = get_dataset(delete_inconcitencies=False, from_disc=False)

#%%
def plot_user_activity_and_met(user, mindate='2013-03-27 04:00:00', maxdate='2013-06-01 3:00:00',  df=None):

    title = f'Actividad del usuario {user}'
    if df is None:
        df = get_dataset(delete_inconcitencies=False, from_disc=False)

    data = get_activity_levels(get_user_data(df, user))
    
    data = data.loc[(data.index.get_level_values(1) >= mindate) &
                    (data.index.get_level_values(1) < maxdate)]
    true_date_range = data.index.get_level_values(1)
    date_range = pd.date_range(mindate, maxdate, freq='h', closed='left')
    none_dates = date_range.difference(true_date_range)
    print('Faltan {0} buckets!'.format(len(none_dates)))
    
    xlabels = mdates.date2num(true_date_range)


    r = data['walkingLevel'].values + data['runningLevel'].values
    w = data['walkingLevel'].values

    plt.close()
    fig, (ax2, ax) = plt.subplots(2, 1, figsize=(15,6), sharex='all')
    fig.suptitle(title, fontsize=16)


    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%A'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=4))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))


    for date in none_dates:
        ax.axvline(mdates.date2num(date))
        ax2.axvline(mdates.date2num(date))


    ax.set_ylim(0, 1)
    ax2.set_ylim(1.3,8)
    ax.set_xlim(xlabels[0], xlabels[-1])


    ax.set_ylabel(f'% de actividad por tipo')
    ax.set_xlabel('Tiempo')
    ax2.set_ylabel('Nivel de MET')

    data = addSedentaryLevel(get_user_data(df, user)).slevel

    data = data.loc[(data.index.get_level_values(1) >= mindate) &
                    (data.index.get_level_values(1) < maxdate)]
    
    for x in [ax, ax2]:
        x.autoscale_view()
        x.grid(True)

    fig.autofmt_xdate()

    ax.fill_between(xlabels, w, 0, facecolor='yellow', alpha=1, label='Caminando')
    ax.fill_between(xlabels, w, r, facecolor='red', alpha=1, label='Corriendo')
    ax.fill_between(xlabels, r, 1, facecolor='black', alpha=.4, label='Estacionario')
    ax.legend(loc='upper right')
    ax2.plot(xlabels, data.values)

    plt.show()
    

# %%
plot_user_activity_and_met(51, '2013-05-22 00:00:00', '2013-05-28 23:59:59', df)

# %%
