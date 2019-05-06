from collections import Counter

import numpy
import numpy as np
import pandas as pd
from haversine import haversine
from scipy.stats.stats import pearsonr
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
pd.options.mode.chained_assignment = None

numpy.random.seed(7)

def createSensingTable(sensor):
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
    df.to_csv('processing/' + sensor + '.csv', index=False)
    return df

def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def get_user_data(data, userId):
    try:
        return data.loc[data.index.get_level_values(0) == userId].copy()
    except KeyError:
        print('El usuario ', userId, ' no existe.')

def get_not_user_data(data, userId):
    try:
        return data.loc[data.index.get_level_values(0) != userId].copy()
    except KeyError:
        print('El usuario ', userId, ' no existe.')

def get_X_y_regression(df):
    dfcopy = df.copy()
    features = [col for col in dfcopy.columns if 'slevel' != col]
    return dfcopy[features].reset_index(drop=True), dfcopy['slevel'].reset_index(drop=True)

def makeSedentaryClasses(df):
    dfcopy = df.copy()
    dfcopy['sclass'] = ''
    dfcopy.loc[df['slevel'] >= 1.5, 'sclass'] = 0.0  # 'sedentary'
    dfcopy.loc[df['slevel'] < 1.5, 'sclass'] = 1.0  # 'not sedentary'
    #dfcopy['actualClass'] = dfcopy['sclass']
    dfcopy.drop(['slevel'], inplace=True, axis=1)
    return dfcopy

def get_X_y_classification(df, withActualClass=True):
    dfcopy = df.copy()
    #if not withActualClass:
    #    dfcopy.drop(['actualClass'], inplace=True, axis=1)
    features = [col for col in dfcopy.columns if 'sclass' != col]
    return dfcopy[features].reset_index(drop=True), dfcopy['sclass'].reset_index(drop=True)

def shift_hours(df, n, columns=None):
    dfcopy = df.copy().sort_index()
    if columns is None:
        columns = df.columns
    for ind, row in dfcopy.iterrows():
        try:
            dfcopy.loc[(ind[0], ind[1]), columns] = dfcopy.loc[(ind[0], ind[1] + pd.DateOffset(hours=n)), columns]
        except KeyError:
            dfcopy.loc[(ind[0], ind[1]), columns] = np.nan
    # print(dfcopy.isna().sum())
    dfcopy.dropna(inplace=True)
    return dfcopy

def create_model(clf):
    numeric_cols = ['numberOfConversations', 'wifiChanges',
                    'silenceLevel', 'voiceLevel', 'noiseLevel',
                    'hourSine','hourCosine',
                    'remainingminutes', 'pastminutes',
                    'distanceTraveled', 'locationVariance']
    transformer = ColumnTransformer([('scale', MinMaxScaler(), numeric_cols)],
                                    remainder='passthrough')
    return make_pipeline(transformer, clf)

def METcalculation(df, metValues=(1.3,5,8.3)):
    dfcopy = df.copy()
    metLevel = (dfcopy['stationaryLevel'] * metValues[0] +
                dfcopy['walkingLevel'] * metValues[1] +
                dfcopy['runningLevel'] * metValues[2])
    dfcopy['slevel'] = metLevel
    return dfcopy

def makeDummies(df):
    dfcopy = df.copy()
    categorical_cols = ['dayofweek', 'activitymajor']
    for col in categorical_cols:
        dfcopy[col] = dfcopy[col].astype('category')
    for col in set(df.columns) - set(categorical_cols):
        dfcopy[col] = dfcopy[col].astype('float')
    dummies = pd.get_dummies(dfcopy.select_dtypes(include='category'))
    dfcopy.drop(categorical_cols, inplace=True, axis=1)
    return pd.concat([dfcopy, dummies], axis=1, sort=False)

def delete_user(df,user):
    return df.copy().loc[df.index.get_level_values(0)!=user]

def get_total_harversine_distance_traveled(x):
    d = 0.0
    samples = x.shape[0]
    for i in np.arange(0,samples):
        try:
            d += haversine(x.iloc[i,:].values, x.iloc[i+1,:].values)
        except IndexError:
            pass
    return d

def delete_sleep_hours(df):
    dfcopy = df.copy()
    return dfcopy.loc[(dfcopy['slevel'] >= 1.5) |
                      ((dfcopy.index.get_level_values(1).hour<22) &
                       (dfcopy.index.get_level_values(1).hour>5))]
#saco horas oscuras

def make_dataset(gran='1h'):
    df = pd.read_pickle('pkl/sedentarismdata_gran{0}.pkl'.format(gran))
    df = delete_user(df, 52)
    df = makeDummies(df)
    df = METcalculation(df)
    pd.to_pickle(df, 'pkl/dataset_gran{0}.pkl'.format(gran))

def get_dataset(gran='1h', with_dummies=False):
    df = pd.read_pickle('pkl/sedentarismdata_gran{0}.pkl'.format(gran))
    df = delete_user(df, 52)
    if with_dummies:
        df = makeDummies(df)
    df = METcalculation(df)
    return df

def series_to_supervised(df, dropnan=True, number_of_lags=None, period=1):
    lags = range(period * number_of_lags, 0, -period)
    columns = df.columns
    n_vars = df.shape[1]
    print(lags,columns,n_vars)
    data, names = list(), list()
    #print('Generating {0} time-lags with period equal {1} ...'.format(number_of_lags, period))
    # input sequence (t-n, ... t-1)
    for i in range(len(lags), 0, -1):
        data.append(shift_hours(df, lags[i-1], df.columns))
        names += [('{0}(t-{1})'.format(columns[j], lags[i-1])) for j in range(n_vars)]
    data.append(df)
    names += [('{0}(t)'.format(columns[j])) for j in range(n_vars)]

    # put it all together
    agg = pd.concat(data, axis=1)
    agg.columns = names
    # drop rows w   ith NaN values
    if dropnan:
        agg.dropna(inplace=True)
    x = agg.iloc[:, 0:number_of_lags* n_vars]
    y = agg.iloc[:, -1]
    return x,y

def make_lagged_datasets(lags=None, period=1, gran='1h'):
    df = pd.read_pickle('pkl/dataset_gran{0}.pkl'.format(gran))
    xs, ys = list(), list()
    for i in df.index.get_level_values(0).drop_duplicates():
        x, y = series_to_supervised(get_user_data(df, i), number_of_lags=lags, period = period)
        xs.append(x)
        ys.append(y)
    x = pd.concat(xs, axis=0)
    y = pd.concat(ys, axis=0)
    df = pd.concat((x,y), axis=1)
    df.to_pickle('pkl/datasets/gran{2}_period{1}_lags{0}.pkl'.format(lags, period,gran))
    del df

def generate_MET_stadistics(df):
    things = list()
    for u in df.index.get_level_values(0).drop_duplicates():
        dfuser = get_user_data(df, u)
        aux = dfuser.droplevel(0).loc[:, 'slevel']
        idx = pd.date_range('2013-03-27 04:00:00', '2013-06-01 3:00:00', freq='h')
        d = pd.DataFrame(index=idx)
        d['slevel'] = aux
        n = d.isna().sum()[0]
        delete_sleep_hours(dfuser)
        dfuser['hourofday'] = dfuser.index.get_level_values(1).hour
        dfuser['dayofweek'] = dfuser.index.get_level_values(1).dayofweek
        stats = dfuser.groupby(['dayofweek', 'hourofday'])['slevel'].agg(['mean', 'std']).dropna()
        corr = pearsonr(stats['mean'], stats['std'])[0]

        things.append([u, stats['mean'].mean(), stats['std'].mean(), corr, n])
        # corrs.append(corr)
    return pd.DataFrame(columns=['user', 'met', 'std', 'corr', 'nb_nulls'], data=things).sort_values('met')

def get_data(personal, lags, period, gran,user=-1):
    data = pd.read_pickle('pkl/datasets/gran{0}_period{1}_lags{2}.pkl'.format(gran,period,lags))
    if personal: return get_user_data(data,user)
    else: return data

def split_x_y(data):
    x = data.iloc[:, [data.columns.get_loc(c) for idx, c in enumerate(data.columns) if c != 'slevel(t)']]
    y = data.iloc[:,data.columns.get_loc('slevel(t)')]
    return x,y

def get_train_test_data(user,standarize, lags, period, gran, personal):

    data = get_data(personal, lags, period, gran, user)
    numeric_cols = ['stationaryLevel', 'walkingLevel', 'runningLevel',
                    'numberOfConversations', 'wifiChanges',
                    'silenceLevel', 'voiceLevel', 'noiseLevel',
                    'hourSine', 'hourCosine',
                    'remainingminutes', 'pastminutes',
                    'locationVariance']
    to_standarize = []
    for col in numeric_cols:
        for lag in range(1, lags + 1):
            to_standarize.append(col + '(t-{0})'.format(lag))
    #aux = to_standarize.copy()
    #aux.append('slevel(t)')
    #data = data.loc[:,aux]
    x,y = split_x_y(data)
    if personal:
        x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, train_size=2/3)
    else:
        x_test = get_user_data(x, user)
        x_train = get_not_user_data(x, user)
        y_test = get_user_data(y, user)
        y_train = get_not_user_data(y, user)
    x_train, y_train = x_train.values.astype("float32"), y_train.values.astype("float32")
    x_test, y_test = x_test.values.astype("float32"), y_test.values.astype("float32")
    if standarize:
        to_standarize = [data.columns.get_loc(c) for c in to_standarize]
        ss = StandardScaler()
        x_train[:, to_standarize] = ss.fit_transform(x_train[:,to_standarize])
        x_test[:, to_standarize] = ss.transform(x_test[:,to_standarize])


    return x_train, y_train, x_test, y_test

#df = pd.read_pickle('pkl/dataset.pkl')
#d = generate_MET_stadistics(df)
'''
if __name__ == '__main__':
    for gran in ['30m']:
        max = 5
        r = range(max, 0, -1)
        for period in [1]:
            for lags in r:
                print('gran',gran,'lags', lags, ' period ', period)
                make_lagged_datasets(lags,period,gran)
'''


