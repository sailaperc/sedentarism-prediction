from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Conv1D, LSTM, Dropout, Dense, Flatten, BatchNormalization, Activation, Input
from sklearn.preprocessing import StandardScaler, PowerTransformer
from keras.models import Sequential
import numpy as np
import pickle
from tcn import compiled_tcn

pd.options.mode.chained_assignment = None


def get_architecture(n):
    model = Sequential()
    input_shape = (time_lags[n], number_of_features)
    if n == 1:
        input_shape = input_shape
        model.add(LSTM(16, input_shape=input_shape, return_sequences=False))
        model.add(Dropout(.8))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mae',
                      optimizer='adam',
                      )
    elif n == 2:
        model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(.6))
        model.add(LSTM(32, input_shape=input_shape, return_sequences=False))
        model.add(Dropout(.6))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mae',
                      optimizer='adam',
                      )
    elif n == 3:
        model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
        model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
        model.add(LSTM(32, input_shape=input_shape, return_sequences=False))
        model.add(Dropout(.6))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dropout(.4))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dropout(.2))
        model.add(Dense(8))
        model.add(Activation('relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mae',
                      optimizer='adam',
                      )
    elif n == 4:
        model = compiled_tcn(return_sequences=False,
                             num_feat=number_of_features,
                             num_classes=0,
                             nb_filters=6,
                             kernel_size=2,
                             dilations=[1, 2, 4],
                             # dilations=[2 ** i for i in range(2, 5)],
                             nb_stacks=1,
                             max_len=time_lags[n],
                             use_skip_connections=True,
                             regression=True,
                             dropout_rate=0.2)
    elif n == 5:
        model = Sequential()
        model.add(Conv1D(32, 2, activation='relu', padding='causal', input_shape=input_shape))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mae',
                      optimizer='adam',
                      )
    elif n == 6:
        model.add(Dense(64, input_dim=29, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(.2))
        model.add(Dense(32, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(.2))
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mae',
                      optimizer='adam',
                      )
    print(model.summary())
    return model


def get_user_data(data, userId):
    try:
        return data.loc[data.index.get_level_values(0) == userId].copy()
    except KeyError:
        print('El usuario ', userId, ' no existe.')


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


def series_to_supervised(df2, dropnan=True, number_of_lags=None):
    lags = range(number_of_lags, 0, -1)
    columns = df2.columns
    n_vars = df2.shape[1]
    cols, names = list(), list()
    print('Generating {0} time-lags...'.format(number_of_lags))
    # input sequence (t-n, ... t-1)
    for i in lags:
        cols.append(shift_hours(df2, i, df.columns))
        names += [('{0}(t-{1})'.format(columns[j], i)) for j in range(n_vars)]
    cols.append(df2)
    names += [('{0}(t)'.format(columns[j])) for j in range(n_vars)]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def train_all():
    ss = StandardScaler()
    for i in users:
        if i == 50:
            ss = PowerTransformer()
        print('Comienzan los entrenamientos con el usuario {0}'.format(i))
        userdata = get_user_data(df, i)
        train_cache[i] = {}
        test_cache[i] = {}
        models[i] = {}
        lags = -1
        for j in range(1, number_of_architectures + 1):

            to_standarize = []

            for col in numeric_cols:
                for lag in range(1, time_lags[j] + 1):
                    to_standarize.append(col + '(t-{0})'.format(lag))

            print('El entrenamiendo del usuario {0} con la aquitectura {1} está por comenzar'.format(i, j))
            if lags != time_lags[j]:
                data = series_to_supervised(userdata, number_of_lags=time_lags[j])
            lags = time_lags[j]
            model = get_architecture(j)

            x = data.iloc[:, 0:time_lags[j] * number_of_features]
            y = data.iloc[:, -1]

            x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, train_size=0.67)
            x_train.loc[:, to_standarize] = ss.fit_transform(x_train[to_standarize])
            x_test.loc[:, to_standarize] = ss.transform(x_test[to_standarize])
            x_train, y_train, x_test, y_test = x_train.values.astype("float32"), y_train.values.astype("float32"), \
                                               x_test.values.astype("float32"), y_test.values.astype("float32")

            if time_lags[j] > 1:
                x_train = x_train.reshape(x_train.shape[0], time_lags[j], number_of_features)
                x_test = x_test.reshape(x_test.shape[0], time_lags[j], number_of_features)
            print('{0} casos de entrenamiento. **** {1} casos para testeo.'.format(x_train.shape[0], x_test.shape[0]))
            history = model.fit(x_train, y_train, epochs=epochs[j], batch_size=batch_size[j],
                                validation_data=(x_test, y_test),
                                verbose=0)

            test_cache[i][j] = {'x_test': x_test, 'y_test': y_test}
            train_cache[i][j] = {'x_train': x_train, 'y_train': y_train}
            models[i][j] = {'model': model, 'history': history}

            print('El entrenamiendo del usuario {0} con la aquitectura {1} ha finalizado'.format(i, j))

df = pd.read_pickle('pkl/dataset.pkl')
numeric_cols = ['stationaryLevel', 'walkingLevel', 'runningLevel',
                'numberOfConversations', 'wifiChanges',
                'silenceLevel', 'voiceLevel', 'noiseLevel',
                'hourSine', 'hourCosine',
                'remainingminutes', 'pastminutes',
                'distanceTraveled', 'locationVariance']
number_of_architectures = 6
users = [50, 31, 4]
batch_size = {1: 64, 2: 64, 3: 64, 4: 64, 5: 64, 6: 64}
time_lags = {1: 8, 2: 12, 3: 12, 4: 8, 5: 4, 6: 1}
epochs = {1: 256, 2: 128, 3: 128, 4: 64, 5: 128, 6: 256}
number_of_features = df.shape[1]

train_cache = {}
test_cache = {}
models = {}
train_all()

pickle.dump(train_cache, open('pkl/train_cache.pkl', 'wb'))
pickle.dump(test_cache, open('pkl/test_cache.pkl', 'wb'))
pickle.dump(models, open('pkl/models.pkl', 'wb'))
