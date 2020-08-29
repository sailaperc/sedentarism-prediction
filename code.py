#%%
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model
from preprocessing.datasets import get_lagged_dataset
from tcn import TCN, tcn_full_summary
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential

#%%
data = get_lagged_dataset(user=32, nb_lags=4)
data = data.values.astype('float64')
X = data[:, :-1]
y = data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, shuffle=False)
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

nb_lags = 4

nb_features = int(X_train.shape[1]/nb_lags)
nb_train_samples = X_train.shape[0]
nb_test_samples = X_test.shape[0]
X_train = X_train.reshape(
    nb_train_samples, nb_lags, nb_features)
X_test = X_test.reshape(
    nb_test_samples, nb_lags, nb_features)


#%%
X_train.shape, X_test.shape, y_train.shape, y_test.shape

#%%
m = Sequential()
m.add(Input(batch_shape=(None, nb_lags, nb_features)))
m.add(TCN(nb_filters=8, dropout_rate=.4, use_batch_norm=True, use_skip_connections=True, use_layer_norm=True, dilations=(1,2,4,8)))
m.add(Dense(1, activation='linear'))
m.compile(optimizer='adam', loss='mse')
#%%
#tcn_full_summary(m, expand_residual_blocks=True)
m.fit(X_train, y_train, epochs=64, verbose=1, batch_size=64, validation_data=(X_test,y_test))
m.evaluate(X_test,y_test)


# %%
import math
import numpy as np
# %%
for l in [244]:
    for k in [3]:
        if l>k:
            print('l', l)
            print('k', k)
            irreal_dilation = math.log(l/k,k)
            print('irreal dilation', irreal_dilation)
            math.log(l/k,k)
            d = max(math.ceil(math.log(l/k,k)+1),1)
            print(f'para {l} lags, {k} kernel_size, {d} dilations y receptive={k**(d-1)*k}')

