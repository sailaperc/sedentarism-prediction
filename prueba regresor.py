import numpy as np
np.random.seed(1337) # for reproducibility
from sklearn.model_selection import train_test_split
from keras.layers import Conv1D, LSTM, Dropout, Dense, Flatten, BatchNormalization, Activation, Input, AveragePooling1D, GlobalAveragePooling1D
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from tcn import compiled_tcn
from utilfunction import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
pd.options.mode.chained_assignment = None
from keras.callbacks import ReduceLROnPlateau
from keras.layers import GaussianNoise, Input

#probar AUTO-keras

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
                             dropout_rate=0.4)
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
        model = Sequential()
        model.add(Dense(100, input_dim=input_shape, activation='tanh'))
        model.add(Dropout(.5))
        model.add(Dense(100, activation='tanh'))
        model.add(Dropout(.5))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mae',
                      optimizer='adam',
                      )
        print(model.summary())
        return model
    #print(model.summary())
    return model


ss = StandardScaler()
to_standarize = []
numeric_cols = ['stationaryLevel', 'walkingLevel', 'runningLevel',
                'numberOfConversations', 'wifiChanges',
                'silenceLevel', 'voiceLevel', 'noiseLevel',
                'hourSine', 'hourCosine',
                'remainingminutes', 'pastminutes',
                'locationVariance']

df = pd.read_pickle('pkl/dataset.pkl')
time_lags = 4
epochs=256
batch=64
user = 57
metric = 'mae'
tresd = True
impersonal = True


for col in numeric_cols:
    for lag in range(1, time_lags + 1):
        to_standarize.append(col + '(t-{0})'.format(lag))

def get_model(input_shape):
    model = Sequential()
    model.add(LSTM(4, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(.8))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mae',
                  optimizer='adam',
                  )
    print(model.summary())
    return model

'''
user 3
perceptron
mae:0.023990275338292122

cnn

rnn

tcn

user 2
perceptron
mae:0.26016202569007874

cnn

rnn

tcn


user 3

percep

cnn

rnn

tcn

user 57

percep
mae:0.4474233388900757

cnn

rnn

tcn

'''


if impersonal:
    data = pd.read_pickle('pkl/dataset_lags{0}.pkl'.format(time_lags)), user
else:
    data = get_user_data(pd.read_pickle('pkl/dataset_lags{0}.pkl'.format(time_lags)), user)

number_of_features = df.shape[1]
input_shape = (time_lags, number_of_features)
x = data.iloc[:, 0:time_lags * number_of_features]
y = data.iloc[:, -1]
if impersonal:
    x_test = get_user_data(user)
    x_train =
    y_test = get_user_data(user)
    y_train =
else:
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, train_size=0.67)
x_train.loc[:, to_standarize] = ss.fit_transform(x_train[to_standarize])
x_test.loc[:, to_standarize] = ss.transform(x_test[to_standarize])
x_train, y_train, x_test, y_test = x_train.values.astype("float32"), y_train.values.astype("float32"), \
                                   x_test.values.astype("float32"), y_test.values.astype("float32")

#estandarizar salida

if tresd:
    x_train = x_train.reshape(x_train.shape[0], time_lags, number_of_features)
    x_test = x_test.reshape(x_test.shape[0], time_lags, number_of_features)
else: input_shape = data.shape[1]-1

print('{0} casos de entrenamiento. **** {1} casos para testeo.'.format(x_train.shape[0], x_test.shape[0]))
model = get_model(input_shape)
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch,
                    validation_data=(x_test, y_test),
                    shuffle=False,
                    verbose=0)

plt.close()
plt.figure(figsize=(15, 4))
plt.plot(y_train, label='Train')
y_pred = model.predict(x_train)
plt.plot(y_pred, label='Predicted')
plt.axhline(y=1.5, color='r', linestyle=':', )
plt.legend(loc='upper right')
plt.show()

#mae = mean_absolute_error(y_train, y_pred)
#print('mae:{0}'.format(mae))

plt.close()
plt.figure(figsize=(15, 4))
plt.plot(y_test, label='Test')
y_pred = model.predict(x_test)
plt.plot(y_pred, label='Predicted')
plt.axhline(y=1.5, color='r', linestyle=':', )
plt.legend(loc='upper right')
plt.show()

plt.close()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print('mae:{0}'.format(mae))



