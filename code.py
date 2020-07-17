#%%
import numpy as np
import pandas as pd
from preprocessing.datasets import get_lagged_dataset, get_dataset
from utils.utils import get_user_data
from preprocessing.studentlife_raw import get_studentlife_dataset, get_sensor_data
from datetime import timedelta, datetime
#%%
def fill_by_interval(df, col):
    tuples = list()
    for _, t in df.iterrows():
        tuples +=  [ (t.userId, d) for d in pd.date_range(start=t['start'], end=t['end'], freq='1h')]

    ind = pd.MultiIndex.from_tuples(tuples, names = ['f','s']).drop_duplicates() 

    aux_series = pd.Series(index=ind, dtype='bool')
    aux_series[:] = True
    s[col] = aux_series
    s.loc[:,col].fillna(False, inplace=True)

def floor_time(df, col='time'):
    df[col] = pd.to_datetime(df[col], unit='s').dt.floor(freq)
    return df

def to_time(df, col='time'):
    df[col] = pd.to_datetime(df[col], unit='s')
    return df

chargedata = get_sensor_data('phonecharge')
chargedata = to_time(chargedata, 'start')
chargedata = to_time(chargedata, 'end')
minutes = 15
freq=f'{minutes}min'
chargedata = chargedata.reset_index().sort_values(by=['userId','start'])

def add_percentage_per_bucket(df):
    to_process_index = (df.end.dt.ceil(freq) - df.start.dt.floor(freq)) > timedelta(seconds=60*minutes)
    while to_process_index.sum() > 0:
        aux = df.loc[to_process_index]
        interval = aux.start.dt.floor(freq) + timedelta(seconds = 60 * minutes)
        aux1 = aux.copy()
        aux1['end'] = interval
        aux2 = aux.copy()
        aux2['start'] = interval 
        #print(df[index].shape)
        #print(df.shape)
        df = pd.concat([df[~to_process_index], aux1, aux2], axis=0).reset_index(drop=True)
        #print(df.shape)
        #print()
        to_process_index = (df.end.dt.ceil(freq) - df.start.dt.floor(freq)) > timedelta(seconds=60*minutes)
        print(to_process_index.sum())


    df['percentage'] = (df.end - df.start) / timedelta(seconds=60*minutes)
    df['time'] = df.start.dt.floor(freq)
    df = df.reset_index().sort_values(by=['userId','start'])
    perc = df.groupby(['userId', 'time'])['percentage'].sum()
    perc.loc[perc>1] = 1.0
    return perc, df
#%%
perc, df = add_percentage_per_bucket(chargedata.copy())

#%%
(df.groupby(['userId', 'time'])['percentage'].count()>1).sum()

