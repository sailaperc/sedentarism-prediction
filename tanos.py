from sklearn.model_selection import train_test_split
from keras.layers import Conv1D, LSTM, Dropout, Dense, Flatten, BatchNormalization, Activation, Input
from sklearn.preprocessing import StandardScaler, PowerTransformer
import pickle
from tcn import compiled_tcn
from utilfunction import *
from tcn import compiled_tcn
import matplotlib.pyplot as plt
from keras.optimizers import Adam, Nadam
from keras.activations import relu, elu
import talos as ta

def smodel(x_train, y_train,x_test, y_test, p):

    model = compiled_tcn(return_sequences=False,
                         num_feat=number_of_features,
                         num_classes=0,
                         nb_filters=p['nb_filters'],
                         kernel_size=p['kernel_size'],
                         dilations=[1, 2, 4],
                         # dilations=[2 ** i for i in range(2, 5)],
                         nb_stacks=1,
                         max_len=lags,
                         regression=True,
                         dropout_rate=p['dropout_rate'])

    history = model.fit(x_train, y_train, epochs=p['epochs'], batch_size=p['batch_size'],
                        validation_data=(x_test, y_test),
                        verbose=2)
    return history, model


df = pd.read_pickle('pkl/dataset.pkl')
number_of_features = df.shape[1]
user = 31
lags = 8
df = pd.read_pickle('pkl/dataset_lags{0}.pkl'.format(lags))
x = df.iloc[:,0:-1]
y = df.iloc[:,-1]

x_train = x.loc[x.index.get_level_values(0)!=user,:]
x_test = x.loc[x.index.get_level_values(0)==user,:]
y_train = y.loc[x.index.get_level_values(0)!=user,:]
y_test = y.loc[x.index.get_level_values(0)==user,:]

to_standarize = []
numeric_cols = ['stationaryLevel', 'walkingLevel', 'runningLevel',
                'numberOfConversations', 'wifiChanges',
                'silenceLevel', 'voiceLevel', 'noiseLevel',
                'hourSine', 'hourCosine',
                'remainingminutes', 'pastminutes',
                'locationVariance']

for col in numeric_cols:
    for lag in range(1, lags + 1):
        to_standarize.append(col + '(t-{0})'.format(lag))

ss = StandardScaler()
x_train.loc[:, to_standarize] = ss.fit_transform(x_train[to_standarize])
x_test.loc[:, to_standarize] = ss.transform(x_test[to_standarize])

x_train, y_train, x_test, y_test = x_train.values.astype("float32"), y_train.values.astype("float32"), \
                                   x_test.values.astype("float32"), y_test.values.astype("float32")

x_train = x_train.reshape(x_train.shape[0], lags, number_of_features)
x_test = x_test.reshape(x_test.shape[0], lags, number_of_features)


p = {'nb_filters' : [6,8],
     'kernel_size':[1, 2, 4],
     'batch_size': [4,128],
     'epochs': [8],
     'dropout_rate': [0,.2,.5],
     }


h = ta.Scan(x_train, y_train, params=p,
            model=smodel,
            dataset_name='sedentarism',
            experiment_no='1',
            x_val=x_test,
            y_val=y_test
            )

