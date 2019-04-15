from collections import Counter
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization,Activation
from haversine import haversine
from keras import backend as K
import numpy

K.tensorflow_backend._get_available_gpus()
numpy.random.seed(7)
import numpy

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

def per_user_regression(df, model):
    print('per_user_regression')
    dfcopy = df.copy()
    mse = []
    for userid in df.index.get_level_values(0).drop_duplicates():
        X, y = get_X_y_regression(get_user_data(dfcopy, userid))
        kfold = StratifiedKFold(n_splits=10)
        results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
        mse.append(-results.mean())
        if userid % 10 == 0:
            print('modelos sobre usuario ', userid, ' finalizado.')
    return mse

def live_one_out_regression(df, model):
    print('live_one_out_regression')
    dfcopy = df.copy()
    mse = []
    i = 0
    logo = LeaveOneGroupOut()
    groups = df.index.get_level_values(0)
    X, y = get_X_y_regression(dfcopy)
    for train_index, test_index in logo.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse.append(mean_squared_error(y_test, y_pred))
        if i % 10 == 0:
            print('modelos sobre usuario ', i, ' finalizado.')
        i += 1
    return mse

def per_user_classification(df, model):
    print('per_user_classification')
    dfcopy = df.copy()
    scoring = ['f1_weighted']
    f1 = []
    kfold = StratifiedKFold(n_splits=10)
    for userid in df.index.get_level_values(0).drop_duplicates():
        X, y = get_X_y_classification(get_user_data(dfcopy, userid))
        results = cross_validate(model, X, y, cv=kfold, scoring=scoring)
        f1.append(results['test_f1_weighted'].mean())
        print('modelos sobre usuario ', userid, ' finalizado.')
    return f1

def live_one_out_classification(df, model):
    dfcopy = df.copy()
    print('live_one_out_classification')
    i = 0
    f1 = []
    logo = LeaveOneGroupOut()
    groups = df.index.get_level_values(0)
    X, y = get_X_y_classification(dfcopy)
    for train_index, test_index in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1.append(f1_score(y_test, y_pred, average='weighted'))
        if i % 10 == 0:
            print('modelos sobre usuario ', i, ' finalizado.')
        i += 1
    return f1

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

def baseline_model():
    estimator = Sequential([
    Dense(256,input_dim=28,kernel_initializer='uniform', kernel_regularizer='l2',use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Dense(128, kernel_initializer='uniform', kernel_regularizer='l2',use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Dense(64, kernel_initializer='uniform',kernel_regularizer='l2',use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Dense(32, kernel_initializer='uniform', kernel_regularizer='l2',use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Dense(1, activation='sigmoid')
    ])
    estimator.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy'])
    return estimator

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

def make_dataset():
    df = pd.read_pickle('pkl/sedentarismdata.pkl')
    df = delete_user(df, 52)
    df = makeDummies(df)
    df = METcalculation(df)
    pd.to_pickle(df, 'pkl/dataset.pkl')

def series_to_supervised(df2, dropnan=True, number_of_lags=None):
    lags = range(number_of_lags, 0, -1)
    columns = df2.columns
    n_vars = df2.shape[1]
    data, names = list(), list()
    print('Generating {0} time-lags...'.format(number_of_lags))
    # input sequence (t-n, ... t-1)
    for i in lags:
        data.append(shift_hours(df2, i, df2.columns))
        names += [('{0}(t-{1})'.format(columns[j], i)) for j in range(n_vars)]
    data.append(df2)
    names += [('{0}(t)'.format(columns[j])) for j in range(n_vars)]

    # put it all together
    agg = pd.concat(data, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    x = agg.iloc[:, 0:number_of_lags* n_vars]
    y = agg.iloc[:, -1]
    return x,y


def make_lagged_datasets(lags=None):
    df = pd.read_pickle('pkl/dataset.pkl')
    xs, ys = list(), list()
    for i in df.index.get_level_values(0).drop_duplicates():
        x, y = series_to_supervised(get_user_data(df, i), number_of_lags=lags)
        xs.append(x)
        ys.append(y)
    x = pd.concat(xs, axis=0)
    y = pd.concat(ys, axis=0)
    df = x.append(y)
    pd.to_pickle(df,'pkl/dataset_lags{0}.pkl'.format(lags))

make_lagged_datasets(2)