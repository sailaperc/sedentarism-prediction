
#%%
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from preprocessing.studentlife_raw import get_sensor_data
from preprocessing.datasets import get_dataset
sns.set_style("whitegrid")
#%%
df = get_sensor_data('gps')
df = df.loc[:,['latitude','longitude', 'time', 'userId']]
df['time'] = pd.to_datetime(df['time'], unit='s')
df_shifted = df.groupby('userId').shift(1)
df_shifted.columns = [f's_{col}' for col in df.columns if col!='userId']
df = pd.concat([df, df_shifted], axis=1)
df.dropna(axis=0, inplace=True)
df['diff_date'] = (df.time - df.s_time).dt.seconds
df['diff_lat'] = df.latitude - df.s_latitude
df['diff_lon'] = df.longitude - df.s_longitude
df['instantaneous_speed'] = np.sqrt( np.square(df.diff_lat / df.diff_date) + 
                            np.square(df.diff_lon / df.diff_date))

df['lat_plus_lon'] = np.sqrt(np.square(df.diff_lat) + np.square(df.diff_lon))
df.time = df.time.dt.floor('1h')
g = df.groupby(['userId','time'])
date_features = g.agg({'instantaneous_speed': ['mean','var'], 'lat_plus_lon': 'sum'})
date_features.columns = ['speed_mean', 'speed_variance','total_distance']
#date_features.fillna(0, inplace=True)


# %%
#date_features.loc[np.isinf(date_features.speed_mean)]
#sns.violinplot(data=date_features, y='speed_variance')
date_features.loc[date_features.speed_variance==0]
#date_features.isnull().sum()
