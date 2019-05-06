from keras.layers import Conv1D, LSTM, Dropout, Dense, Flatten, BatchNormalization, Activation, Input
from sklearn.linear_model import LinearRegression
import pickle
from tcn import compiled_tcn
from utils import *
from keras import Sequential

np.random.seed(1337) # for reproducibility

pd.options.mode.chained_assignment = None

def get_architecture(n):
    model = Sequential()
    input_shape = (time_lags[n], number_of_features)
    if n == 1:
        model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(.6))
        model.add(LSTM(32, input_shape=input_shape, return_sequences=False))
        model.add(Dropout(.6))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mae',
                      optimizer='adam',
                      )
    elif n == 2:
        model = compiled_tcn(return_sequences=False,
                             num_feat=number_of_features,
                             num_classes=0,
                             nb_filters=5,
                             kernel_size=2,
                             dilations=[1, 2],
                             # dilations=[2 ** i for i in range(2, 5)],
                             nb_stacks=1,
                             max_len=time_lags[n],
                             use_skip_connections=True,
                             regression=True,
                             dropout_rate=0.4)
    elif n == 3:
        model = Sequential()
        model.add(Conv1D(32, 2, activation='relu', padding='causal', input_shape=input_shape))
        model.add(Dropout(0.7))
        model.add(Flatten())
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mae',
                      optimizer='adam',
                      )
    elif n==4:
        model.add(Dense(64, input_dim=number_of_features, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(.4))
        model.add(Dense(32, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(.4))
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mae',
                      optimizer='adam',
                      )
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

            print('El entrenamiendo del usuario {0} con la aquitectura {1} está por comenzar'.format(i, j))

            lags = time_lags[j]
            model = get_architecture(j)

            x_train, y_train, x_test, y_test = get_train_test_data(i,True,lags,1,'1h',True)
            if j!=4:
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

'''
def train_baseline():
    for i in users:
        x_train, y_train, x_test, y_test = get_train_test_data(i, True, 1, 1, '1h', True)
        model = LinearRegression()
        model.fit(x_train,y_train)
        y_pred = model.predict(x_train)
        e_train = mean_squared_error(y_train, y_pred)
        y_pred = model.predict(x_test)
        e_test = mean_squared_error(y_test, y_pred)
        print('Usuario {0} '.format(i), 'mae (train): ', e_train, '- mae (test): ', e_test)
'''

data = get_data(True,1,1,'1h',56)

numeric_cols = ['stationaryLevel', 'walkingLevel', 'runningLevel',
                'numberOfConversations', 'wifiChanges',
                'silenceLevel', 'voiceLevel', 'noiseLevel',
                'hourSine', 'hourCosine',
                'remainingminutes', 'pastminutes',
                'locationVariance']
number_of_architectures = 4
users = [50, 31, 4]
batch_size = {1: 64, 2: 64, 3: 64, 4:64}
time_lags = {1: 4,  2: 4, 3: 4, 4:1}
epochs = {1: 128, 2: 128, 3: 128, 4:256}
number_of_features = data.shape[1]-1

train_cache = {}
test_cache = {}
models = {}
train_all()

pickle.dump(train_cache, open('pkl/train_cache.pkl', 'wb'))
pickle.dump(test_cache, open('pkl/test_cache.pkl', 'wb'))
pickle.dump(models, open('pkl/models.pkl', 'wb'))
