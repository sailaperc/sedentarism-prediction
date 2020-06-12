from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessing.studentlife_raw import get_sensor_data
sns.set(color_codes=True)
sns.set_style("whitegrid", {'axes.grid' : False})
sensor_data_files = ['activity', 'audio', 'gps', 'dark',
                         'phonelock', 'wifi', 'phonecharge',
                         'calendar', 'wifi_location', 'conversation']

def plot_inference_info():
    df = get_sensor_data('activity')
    df.columns = ['time', 'inference', 'userId']
    df = df.loc[df['inference'] == 3]
    min_date = datetime.fromtimestamp(min(df.time))
    min_date = min_date.replace(minute=0, second=0, microsecond=0)
    max_date = datetime.fromtimestamp(max(df.time))
    max_date = max_date.replace(minute=0, second=0, microsecond=0)
    print(min_date)
    s = pd.DataFrame(index=pd.MultiIndex.from_product(iterables=[df['userId'].drop_duplicates(),
                                                                 pd.date_range(min_date,
                                                                               max_date,
                                                                               freq='1h')],
                                                      names=['userId', 'time']))
    print(s)
    df['time'] = pd.to_datetime(df['time'], unit='s').dt.floor('1h')
    grouped_per_hour_and_user = df.groupby(['userId', 'time'])['inference'].count()
    print(grouped_per_hour_and_user)
    s.loc[:, 'inference'] = grouped_per_hour_and_user
    s = s.fillna(0)
    vals = s.to_numpy()



    plt.close()
    plt.title('Registros desconocidos')
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(vals)
    ax.set_xlabel('Horas')
    ax.set_ylabel('Numero de valores desconocidos')

    ax2 = fig.add_subplot(122)
    vals = np.cumsum(vals)
    ax2.plot(vals)
    ax2.set_xlabel('Horas')
    ax2.set_ylabel('Numero de valores desconocidos')

    plt.subplots_adjust()


    fig.show()

def print_basic_info(sensor):
    '''
    plots info of activity data
    :return:
    '''
    print(f'basic info for {sensor}')
    df = get_sensor_data(sensor)
    df.columns = ['time', 'inference', 'userId']
    nb_regs = df.shape[0]
    print(f'Cantidad total de registros: {nb_regs}')

    '''
    nb_unknown = df.loc[df["inference"]==3].shape
    print(f'Cantidad de unknowns: {nb_unknown}')
    min_date = datetime.fromtimestamp(min(df.time))
    print(f'min date: {min_date}')
    max_date = datetime.fromtimestamp(max(df.time))
    print(f'max date: {max_date}')
    '''
    del df

def print_frec_info(sensor, freq):
    print(f'freq info for {sensor} and {freq}')
    df = get_sensor_data(sensor)
    df.columns = ['time', 'inference', 'userId']
    df = df.loc[df['inference'] != 3]
    min_date = datetime.fromtimestamp(min(df.time))
    max_date = datetime.fromtimestamp(max(df.time))
    s = pd.DataFrame(index=pd.MultiIndex.from_product(iterables=[df['userId'].drop_duplicates(),
                                                                         pd.date_range(min_date,
                                                                                       max_date,
                                                                                       freq=freq)],
                                                              names=['userId', 'time']))

    print(f'Total number of 1h blocks : {s.shape[0]}')

    df['time'] = pd.to_datetime(df['time'], unit='s').dt.floor(freq)
    grouped_per_hour_and_user = df.groupby(['userId', 'time'])['inference'].count()
    print(f'Total number of {freq} blocks with data available: {grouped_per_hour_and_user.shape[0]}')
    vals = grouped_per_hour_and_user.to_numpy()
    print(f'Average and std available logs for {freq}: {vals.mean()} / {vals.std()}')
    del vals
    del df
    del grouped_per_hour_and_user

def print_all_info():
    for t in ['activity', 'audio']:
        print_basic_info(t)
        for f in ['1h','30min']:
            print_frec_info(t, f)

'''

s.loc[:, 'activityId'] = grouped_per_hour_and_user


# Prints a histogram over the number of logs available per hour and user
vals = grouped_per_hour_and_user.to_numpy()
sns.distplot(vals, kde=False, bins=1000)
#plt.xlim(0,100)
plt.ylim(0,50)
plt.ylabel('Nro. de registros')
plt.xlabel('Data')
plt.show()


plt.close()
vals = s['activityId'].to_numpy()
plt.plot(vals)
plt.show()
'''


plot_inference_info()

