'''

#Audio Inference
#ID	Description
#0	Silence
#1	Voice
#2	Noise
#3	Unknown

#Activity Inference ID	Description
#0	Stationary
#1  Walking
#2	Running
#3	Unknown

cuando la actividad que mas se lleva a cabo es unknown, el porcentaje de actividad
sedentaria para esa hora es similar al promedio de actividad sedentaria para las
hora donde la act q mas se lleva a cabo es 1 (walking), por eso se lo va a tomar
como si fuera de ese tipo

activitymajor
0    0.937012
1    0.296808
2    0.073199
3    0.201710


#
#
# ## Feature generation ##
#
# **Features:**
# * Stationaty mean per hour
# * Day of the week (weekday,saturday or sunday)
# * Hour of the day
# * activityMajor: the type of activity with the most instances in a 1-hour time bucket
# * audioMajor
# * latitude average and stv
# * longitud avg and stv
# * is Charging

#como tratar los valores nulos??
    # los deadlines son solo de 44 de 49 estudiantes
    # los datos de audio no estan para todas la horas de todos los estudiantes
    # los datos de ubicacion son infimos


hay 14420.575126 en promedio de muestros de actividad por hora
'''
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from collections import Counter
from utils.utils import file_exists
from datetime import datetime

'''
def get_total_harversine_distance_traveled(x):
    d = 0.0
    samples = x.shape[0]
    for i in np.arange(0, samples):
        try:
            d += haversine(x.iloc[i, :].values, x.iloc[i + 1, :].values)
        except IndexError:
            pass
    return d
'''


def create_sensing_table(sensor):
    """
    Creates one dataframe from all the sensor data of all users

    dataset raw data should be at dataset/sensing/ in the project folder
    """
    filename = f'pkl/sensing_data/{sensor}.pkl'
    if file_exists(filename):
        print('sensing_data already generated')
    else:
        path = 'dataset/sensing/' + sensor + '/' + sensor + '_u'
        df = pd.read_csv(path + '00' + '.csv', index_col=False)
        df['userId'] = '00'
        for a in range(1, 60):
            userId = '0' + str(a) if a < 10 else str(a)
            try:
                aux = pd.read_csv(path + userId + '.csv', index_col=False)
                aux['userId'] = a
                df = df.append(aux)
            except:
                pass
        df.to_pickle(filename)


def create_sensing_tables():
    sensor_data_files = ['activity', 'audio', 'gps', 'dark',
                         'phonelock', 'wifi', 'phonecharge',
                         'calendar', 'wifi_location', 'conversation']
    for file in sensor_data_files:
        create_sensing_table(file)


def get_sensor_data(sensor) -> pd.DataFrame:
    return pd.read_pickle(f'pkl/sensing_data/{sensor}.pkl')


def get_studentlife_dataset(freq='1h'):
    def Most_Common(lst):
        data = Counter(lst)
        return data.most_common(1)[0][0]

    filename = f'pkl/sedentarismdata_gran{freq}.pkl'
    if not file_exists(filename):
        print(f'{filename} does not exist. This may take a while...')
        # prepare activity data
        sdata = get_sensor_data('activity')
        sdata.columns = ['time', 'activityId', 'userId']
        sdata = sdata.loc[sdata['activityId'] != 3]
        min_date = datetime.fromtimestamp(min(sdata.time))
        max_date = datetime.fromtimestamp(max(sdata.time))

        sdata['time'] = pd.to_datetime(sdata['time'], unit='s').dt.floor(freq)

        '''
        Set dataset index from the cartesian product between
        the users id and
        the minimun and maximun date found

        '''
        s = pd.DataFrame(index=pd.MultiIndex.from_product(iterables=[sdata['userId'].drop_duplicates(),
                                                                     pd.date_range(min_date,
                                                                                   max_date,
                                                                                   freq=freq)],
                                                          names=['userId', 'time']))

        sdata = pd.concat([sdata, pd.get_dummies(sdata['activityId'], prefix='act')], axis=1, sort=False)

        # logs per activity
        grouped = sdata.groupby(['userId', 'time'])
        s.loc[:, 'stationaryLevel'] = grouped['act_0'].mean()
        s.loc[:, 'walkingLevel'] = grouped['act_1'].mean()
        s.loc[:, 'runningLevel'] = grouped['act_2'].mean()
        s.dropna(how='all', inplace=True)

        # 2013-03-27 04:00:00
        # 2013-06-01 3:00:00

        # sedentary mean
        # hourofday

        hours = s.index.get_level_values('time').hour
        s['hourSine'] = np.sin(2 * np.pi * hours / 23.0)
        s['hourCosine'] = np.cos(2 * np.pi * hours / 23.0)

        # dayofweek
        s['dayofweek'] = s.index.get_level_values('time').dayofweek

        # activitymajor
        s['activitymajor'] = sdata.groupby(['userId', 'time'])['activityId'].apply(Most_Common)

        s['pastminutes'] = s.index.get_level_values(1).hour * 60 + s.index.get_level_values(1).minute
        s['remainingminutes'] = 24 * 60 - s['pastminutes']

        # prepare audio data
        adata = get_sensor_data('audio')
        adata.columns = ['time', 'audioId', 'userId']
        adata['time'] = pd.to_datetime(adata['time'], unit='s')
        adata['time'] = adata['time'].dt.floor(freq)

        # audiomajor
        # los siguientes usuarios poseen horas completas en las cuales no tienen ningun registro de audio
        # s[s['audiomajor'].isna()].groupby('userId')['audiomajor'].count()
        # s['audiomajor'] = np.NaN
        # s['audiomajor'] = adata.groupby(['userId', 'time'])['audioId'].apply(Most_Common).astype('int')

        # 0	Silence
        # 1	Voice
        # 2	Noise
        # 3	Unknow

        adata = pd.concat([adata, pd.get_dummies(adata['audioId'], prefix='act')], axis=1, sort=False)

        # logs per activity
        s.loc[:, 'silenceLevel'] = adata.groupby(['userId', 'time'])['act_0'].mean()
        s.loc[:, 'voiceLevel'] = adata.groupby(['userId', 'time'])['act_1'].mean()
        s.loc[:, 'noiseLevel'] = adata.groupby(['userId', 'time'])['act_2'].mean()
        s.fillna(0, inplace=True)

        # latitude and longitude mean and std
        gpsdata = get_sensor_data('gps')
        gpsdata['time'] = pd.to_datetime(gpsdata['time'], unit='s')
        gpsdata['time'] = gpsdata['time'].dt.floor(freq)

        # gpsdata.loc[gpsdata['travelstate'].isna() & gpsdata['speed']>0, 'travelstate'] = 'moving'
        # gpsdata.loc[gpsdata['travelstate'].isna(), 'travelstate'] = 'stationary'
        # kmeans = cluster.KMeans(15)
        # kmeans.fit(gpsdata[['latitude', 'longitude']].values)
        # gpsdata['place'] = kmeans.predict(gpsdata[['latitude', 'longitude']])
        # s['place'] = gpsdata.groupby(['userId', 'time'])['place'].apply(Most_Common)

        # s['distanceTraveled'] = gpsdata.groupby( by= ['userId', pd.Grouper(key='time', freq='H')])['latitude','longitude'].\
        #   apply(get_total_harversine_distance_traveled)
        # s['distanceTraveled'].fillna(0, inplace=True)

        s['locationVariance'] = gpsdata.groupby(['userId', 'time'])['longitude'].std() \
                                + gpsdata.groupby(['userId', 'time'])['latitude'].std()
        s['locationVariance'].fillna(0, inplace=True)
        # calculo la distancia total recorrida por el usuario en una hora

        # prepare charge data
        chargedata = get_sensor_data('phonecharge')
        chargedata['start'] = pd.to_datetime(chargedata['start'], unit='s').dt.floor(freq)
        chargedata['end'] = pd.to_datetime(chargedata['end'], unit='s').dt.floor(freq)

        # isCharging
        s['isCharging'] = False
        for index, t in chargedata.iterrows():
            for date in pd.date_range(start=t['start'], end=t['end'], freq=freq):
                try:
                    s.loc[[(t['userId'], date)], 'isCharging'] = True
                except KeyError:
                    pass

        # prepare lock data
        lockeddata = get_sensor_data('phonelock')
        lockeddata['start'] = pd.to_datetime(lockeddata['start'], unit='s').dt.floor(freq)
        lockeddata['end'] = pd.to_datetime(lockeddata['end'], unit='s').dt.floor(freq)

        # isLocked
        s['isLocked'] = False
        for index, t in lockeddata.iterrows():
            for date in pd.date_range(start=t['start'], end=t['end'], freq=freq):
                try:
                    s.loc[[(t['userId'], date)], 'isLocked'] = True
                except KeyError:
                    pass

        # prepare dark data
        darkdata = get_sensor_data('sensing_data/dark.csv')
        darkdata['start'] = pd.to_datetime(darkdata['start'], unit='s').dt.floor(freq)
        darkdata['end'] = pd.to_datetime(darkdata['end'], unit='s').dt.floor(freq)

        # isInDark
        s['isInDark'] = False
        for index, t in darkdata.iterrows():
            for date in pd.date_range(start=t['start'], end=t['end'], freq=freq):
                try:
                    s.loc[[(t['userId'], date)], 'isInDark'] = True
                except KeyError:
                    pass

        # prepare conversation data
        conversationData = get_sensor_data('conversation')
        conversationData['start_timestamp'] = pd.to_datetime(conversationData['start_timestamp'], unit='s').dt.floor(
            freq)
        conversationData[' end_timestamp'] = pd.to_datetime(conversationData[' end_timestamp'], unit='s').dt.floor(freq)

        s['numberOfConversations'] = 0
        for index, t in conversationData.iterrows():
            if t['start_timestamp'] == t[' end_timestamp']:
                try:
                    s.loc[[(t['userId'], t['start_timestamp'])], 'numberOfConversations'] += 1
                except KeyError:
                    pass
            else:
                dates = pd.date_range(start=t['start_timestamp'], end=t[' end_timestamp'], freq=freq)
                for date in pd.date_range(start=t['start_timestamp'], end=t[' end_timestamp'], freq=freq):
                    try:
                        s.loc[[(t['userId'], date)], 'cantConversation'] += 1
                    except KeyError:
                        pass

        '''
        #cargo los datos de deadlines
        deadlines = get_sensor_data('deadlines').iloc[:, 0:72]
        deadlines = pd.melt(deadlines, id_vars='uid', var_name='time', value_name='exams')
        deadlines['time'] = pd.to_datetime(deadlines['time'])
        deadlines['uid'] = deadlines['uid'].str.replace('u', '', regex=True).astype('int')
        deadlines = deadlines.loc[deadlines['exams'] > 0]
        deadlines = deadlines.set_index('uid')
    
    
        a = pd.to_datetime(max(deadlines['time']), yearfirst=True)
        b = pd.to_datetime(min(deadlines['time']), yearfirst=True)
        maxTime = int((a-b).total_seconds()/3600)
    
        #beforeNextDeadline
        s['beforeNextDeadline'] = 0
        def getHourstoNextDeadLine(user, date):
            try: #para usuarios sobre los que no hay datos de examenes
                possibledeadlines = deadlines.loc[user, 'time']
                possibledeadlines = possibledeadlines[possibledeadlines >= date.floor('h')]
                if not possibledeadlines.empty: #para cuando no hay mas fechas de examenes
                    deadline = min(possibledeadlines)
                    if date.floor('h') == deadline:
                        return 0
                    else:
                        diff = int((deadline - date).total_seconds()/3600)
                    return diff
                return maxTime
            except KeyError:
                return maxTime
    
    
        for ind, row in s.iterrows():
            s.at[ind, 'beforeNextDeadline'] = getHourstoNextDeadLine(ind[0], pd.to_datetime(ind[1]))
    
        #afterLastDeadline
        s['afterLastDeadline'] = 0
        def getHourstoNextDeadLine(user, date):
            try: #para usuarios sobre los que no hay datos de examenes
                possibledeadlines = deadlines.loc[user, 'time']
                possibledeadlines = possibledeadlines[possibledeadlines < date.floor('h')]
                if not possibledeadlines.empty: #para cuando no hay mas fechas de examenes
                    deadline = max(possibledeadlines)
                    if date.floor('h') == deadline:
                        return 0
                    else:
                        diff = int((date - deadline).total_seconds()/3600)
                    return diff
                return maxTime
            except KeyError:
                return maxTime
    
    
        for ind, row in s.iterrows():
            s.at[ind, 'afterLastDeadline'] = getHourstoNextDeadLine(ind[0], pd.to_datetime(ind[1]))
        '''

        calendardata = get_sensor_data('calendar')
        calendardata['time'] = pd.to_datetime(calendardata['DATE'] + ' ' + calendardata['TIME'])
        calendardata['time'] = calendardata['time'].dt.floor(freq)
        calendardata = calendardata.set_index(['userId', 'time'])

        s['hasCalendarEvent'] = False
        s.loc[s.index & calendardata.index, 'hasCalendarEvent'] = True

        # hay datos sobre los wifi mas cercano y ademas sobre los que el usuario estuvo
        # dentro del lugar dnd estaba el wifi,
        # hasta elmomento no se utilizan los datos de wifi cercanos
        # se deja el wifi mas cercano, ademas de la cantidad de wifis
        # a los que se conecto cada usuario en una hora, que puede ser un indicador
        # de sedentarismo

        wifidata = get_sensor_data('wifi_location')
        wifidata['time'] = pd.to_datetime(wifidata['time'], unit='s').dt.floor(freq)
        wifidataIn = wifidata.loc[wifidata['location'].str.startswith('in')]
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(wifidataIn['location'].values)
        wifidataIn['location'] = integer_encoded

        # s['wifiMajor'] = 0.0
        # s['wifiMajor'] = wifidataIn.groupby(['userId', 'time'])['location'].apply(Most_Common)
        # s.loc[s['wifiMajor'].isna()] = 0
        wifidataIn.reset_index(inplace=True, drop=True)

        def funct(x):
            changes = 1
            last = x.iloc[0]
            for v in x:
                if v != last:
                    changes += 1
                last = v
            return changes

        s['wifiChanges'] = wifidataIn.groupby(['userId', 'time'])['location'].apply(funct)
        s.loc[s['wifiChanges'].isna(), 'wifiChanges'] = 0

        # a = wifidataIn.groupby(['userId', 'time'])['location']
        # wifidataNear = wifidata.loc[wifidata['location'].str.startswith('near')]

        s.to_pickle(filename)
    else:
        print('Prepocessed StudentLife dataset already generated!')

    return pd.read_pickle(filename)
