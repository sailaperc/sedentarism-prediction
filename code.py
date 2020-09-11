#%%
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model
from preprocessing.datasets import get_lagged_dataset
from tcn import TCN, tcn_full_summary
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, MaxPooling1D
from tensorflow.keras.layers import Conv1D, Dense, Flatten, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout, SpatialDropout1D

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
m.add(Input(batch_shape=(None, nb_lags,nb_features)))
m.add(Conv1D(64,2, padding='same'))
m.add(SpatialDropout1D(.6))
m.add(Conv1D(64,2, padding='same'))
m.add(SpatialDropout1D(.6))
m.add(Conv1D(64,2, padding='same'))
m.add(SpatialDropout1D(.6))
m.add(Flatten())
# m.add(Dense(64))
# m.add(BatchNormalization())
# m.add(Dropout(.6))
m.add(Dense(1))
m.compile(Adam(), loss='MSE')
m.summary()
m.fit(X_train, y_train, epochs=256, verbose=2, batch_size=32, validation_data=(X_test,y_test))
m.evaluate(X_test,y_test)

# %%
