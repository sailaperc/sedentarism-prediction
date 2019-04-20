from sklearn.model_selection import train_test_split
from keras.layers import Conv1D, LSTM, Dropout, Dense, Flatten, BatchNormalization, Activation, Input
from sklearn.preprocessing import StandardScaler, PowerTransformer
import pickle
from tcn import compiled_tcn
from utilfunction import *
import os

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
        model.add(Dense(64, input_dim=number_of_features, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(.2))
        model.add(Dense(32, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(.2))
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mae',
                      optimizer='adam',
                      )
    #print(model.summary())
    return model


def train_all():
    ss = StandardScaler()
    for i in users:
        print('Comienzan los entrenamientos con el usuario {0}'.format(i))
        train_cache[i] = {}
        test_cache[i] = {}
        models[i] = {}
        for j in range(1, number_of_architectures + 1):

            to_standarize = []

            for col in numeric_cols:
                for lag in range(1, time_lags[j] + 1):
                    to_standarize.append(col + '(t-{0})'.format(lag))
            print('El entrenamiendo del usuario {0} con la aquitectura {1} está por comenzar'.format(i, j))

            lags = time_lags[j]
            data = get_user_data(pd.read_pickle('pkl/dataset_lags{0}.pkl'.format(lags)), i)
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
                'locationVariance']
number_of_architectures = 6
users = [3, 2, 57]
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
