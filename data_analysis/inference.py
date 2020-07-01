from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessing.studentlife_raw import get_sensor_data, get_studentlife_dataset
from preprocessing.datasets import get_dataset
sns.set_style("whitegrid")

def plot_activity_logs_per_user(only_unknowns=True):
    df = get_sensor_data('activity')
    df.columns = ['time', 'inference', 'userId']
    df['time'] = pd.to_datetime(df['time'], unit='s').dt.floor('1h')

    df_all = df.groupby(['userId','time'], as_index=False).size().to_frame('count')
    df_all.reset_index(inplace=True)

    if only_unknowns:
        df = df[df['inference'] == 3]
        title = 'Registros de actividad desconocidos por usuario'
    else:
        df = df[df['inference'] != 3]
        title = 'Registros de actividad no desconocidos por usuario'

    df = df.groupby(['userId','time']).size().to_frame('count')
    df.reset_index(inplace=True)

    plt.close()
    fig, (ax,ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex='all')
    sns.barplot(x="userId", y="count", data=df, ax=ax,
                estimator=np.mean, ci=None,
                palette=sns.color_palette("Paired", 10))
    plt.xticks(rotation='vertical')
    ax.set_xlabel("")
    ax.set_ylabel("Media")
    ax.set_title(title)

    sns.barplot(x="userId", y="count", data=df, ax=ax2,
                estimator=np.std, ci=None,
                palette=sns.color_palette("Paired", 10))
    ax2.set_ylabel("Desviación estándar")
    ax2.set_xlabel("User ID")
    plt.show()


def plot_activity_unknown_cumsum():
    df = get_sensor_data('activity')
    df.columns = ['time', 'inference', 'userId']
    df = df[df['inference'] == 3]
    min_date = datetime.fromtimestamp(df.time.min())
    min_date = min_date.replace(minute=0, second=0, microsecond=0)
    max_date = datetime.fromtimestamp(df.time.max())
    max_date = max_date.replace(minute=0, second=0, microsecond=0)
    print(min_date)
    s = pd.DataFrame(index=pd.MultiIndex.from_product(iterables=[df['userId'].drop_duplicates(),
                                                                 pd.date_range(min_date,
                                                                               max_date,
                                                                               freq='1h')],
                                                      names=['userId', 'time']))
    print(s)
    df['time'] = pd.to_datetime(df['time'], unit='s').dt.floor('1h')
    df_size = df.groupby(['userId', 'time']).size()
    print(df_size)
    s.loc[:, 'inference'] = df_size
    s = s.fillna(0)
    vals = s.to_numpy()
    vals = np.cumsum(vals)

    plt.close()
    fig = plt.figure()
    # ax = fig.add_subplot(121)
    # ax.plot(vals)
    # ax.set_xlabel('Horas')
    # ax.set_ylabel('Suma acumulativa de valores desconocidos')

    ax2 = fig.add_subplot(111)
    ax2.plot(vals)
    ax2.set_xlabel('Buckets de una hora')
    ax2.set_ylabel('Registros desconocidos')
    ax2.set_title('Suma acumulativa de registros desconocidos por hora')
    plt.subplots_adjust()

    fig.show()


def print_basic_info(sensor):
    '''
    plots info of sensor data such as :
    - nb_reg
    :return:
    '''
    print(f'basic info for {sensor}')
    df = get_sensor_data(sensor)
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


def print_tsd_info(sensor, freq):
    '''
    print info related to the frequency of the sensor data

    :param sensor:
    :param freq:
    :return:
    '''
    print(f'freq info for {sensor} and {freq}')
    df = get_sensor_data(sensor)

    if 'time' in df.columns:
        time = 'time'
    else: time = 'timestamp'

    df = df.loc[df[' activity inference'] != 3]

    min_date = datetime.fromtimestamp(min(df[time]))
    max_date = datetime.fromtimestamp(max(df[time]))
    s = pd.DataFrame(index=pd.MultiIndex.from_product(iterables=[df['userId'].drop_duplicates(),
                                                                         pd.date_range(min_date,
                                                                                       max_date,
                                                                                       freq=freq)],
                                                              names=['userId', time]))
    print(f'Total number of 1h blocks : {s.shape[0]}')

    df[time] = pd.to_datetime(df[time], unit='s').dt.floor(freq)
    grouped_per_hour_and_user = df.groupby(['userId', time]).size()
    print(f'Total number of {freq} blocks with data available: {len(grouped_per_hour_and_user)}')
    vals = grouped_per_hour_and_user.to_numpy()
    print(f'Average and std available logs for {freq}: {vals.mean()} / {vals.std()}')
    print(f'Min and max available logs for {freq}: {vals.min()} / {vals.max()}')

    del vals
    del df
    del grouped_per_hour_and_user


def print_tsi_info(sensor, freq):
    '''
    print info related to the frequency of the sensor data

    :param sensor:
    :param freq:
    :return:
    '''
    print(f'freq info for {sensor} and {freq}')
    df = get_sensor_data(sensor)

    if 'start' in df.columns:
        start = 'start'
        end = 'end'
    else:
        start = 'start_timestamp'
        end = ' end_timestamp'

    df['diff'] = df[end] - df[start]

    print(f'Promedio y desviación estándar del tamaño de los intervalos: '
          f'{ round(df["diff"].mean() / 3600, 3) } / {round(df["diff"].std() / 3600, 3) }')
    print(f'Maximo y minimo del tamaño de los intervalos: '
          f'{round(df["diff"].min() / 3600, 3)} / {round(df["diff"].max() / 3600, 3)}')


def print_all_info():

    TSD = ['activity', 'audio', 'gps', 'bt',
           'wifi', 'calendar', 'wifi_location']
    TSI = ['dark', 'phonelock',
            'phonecharge', 'conversation']

    # for t in TSD:
    #     print_basic_info(t)
    #     for f in ['1h','30min']:
    #         print_tsd_info(t, f)

    for t in TSI:
        print_basic_info(t)
        for f in ['1h','30min']:
            print_tsi_info(t, f)


def plot_portion_of_activity(only_unknowns=True):
    df = get_sensor_data('activity')
    df.columns = ['time', 'inference', 'userId']
    df['time'] = pd.to_datetime(df['time'], unit='s').dt.floor('1h')

    df_all = df.groupby(['userId','time'], as_index=False).size().to_frame('count')
    df_all.reset_index(inplace=True)

    if only_unknowns:
        df = df[df['inference'] == 3]
        title = 'Registros de actividad desconocidos por usuario'
    else:
        df = df[df['inference'] != 3]
        title = 'Registros de actividad no desconocidos por usuario'

    df = df.groupby(['userId','time']).size().to_frame('count')
    df.reset_index(inplace=True)

    plt.close()
    fig, (ax,ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex='all')
    sns.barplot(x="userId", y="count", data=df, ax=ax,
                estimator=np.mean, ci=None,
                palette=sns.color_palette("Paired", 10))
    plt.xticks(rotation='vertical')
    ax.set_xlabel("")
    ax.set_ylabel("Media")
    ax.set_title(title)

    sns.barplot(x="userId", y="count", data=df, ax=ax2,
                estimator=np.std, ci=None,
                palette=sns.color_palette("Paired", 10))
    ax2.set_ylabel("Desviación estándar")
    ax2.set_xlabel("User ID")
    plt.show()


def plot_met_level_stads():
    df = get_studentlife_dataset()

df.isnull().sum()



