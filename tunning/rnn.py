#%%
from experiments.Experiment import PersonalExperiment, ImpersonalExperiment
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import Reshape, MaxPooling2D
from tensorflow.keras.layers import Conv2D, Dense, Flatten, LSTM
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
#from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args

from preprocessing.datasets import get_lagged_dataset
from utils.utils import get_user_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math


dim_learning_rate = Integer(low=-6, high=-2, name='learning_rate')
dim_num_lstm_layers = Integer(low=1, high=2, name='num_lstm_layers')
dim_num_lstm_nodes = Integer(low=2, high=9, name='num_lstm_nodes')
dim_dropout = Real(low=.0, high=.8, name='dropout')
dim_num_epochs = Integer(low=2, high=6, name='num_epochs')

dimensions = [dim_learning_rate,
              dim_num_lstm_layers,
              dim_num_lstm_nodes,
              dim_dropout,
              dim_num_epochs]

default_parameters = [-5, 1, 4, .3, 4]


# def log_dir_name(learning_rate,
#                  num_lstm_layers, num_lstm_nodes, dropout, num_epochs):

#     s = "./logs/lr_{0:.0e}_nodes_{1}_{2}_{3}_{4}/"
#     log_dir = s.format(learning_rate,
#                        num_lstm_layers,
#                        num_lstm_nodes,
#                        num_epochs,
#                        dropout
#                        )
#     return log_dir


def create_model_fn(learning_rate, num_lstm_layers, num_lstm_nodes, dropout):
    #print(learning_rate, num_lstm_layers, num_lstm_nodes, dropout)
    def create_model():
        use_two_layers = num_lstm_layers == 2

        model = Sequential(name='rnn')
        #model.add(InputLayer(input_shape=(input_shape,)))

        name = 'layer_LSTM_{0}'.format(str(2**num_lstm_nodes))
        model.add(LSTM(2**num_lstm_nodes,
                    return_sequences=use_two_layers,
                    name=name))
        model.add(Dropout(dropout))

        if use_two_layers:
            model.add(LSTM(2**(num_lstm_nodes-1), return_sequences=False))

        model.add(Dense(1, activation='linear'))
        optimizer = Adam(lr=learning_rate)
        model.compile(optimizer=optimizer,
                    loss='MSE',
                    metrics=keras.metrics.MSE)
        #model.summary()
        return model
    return create_model

path_best_model = 'best_model.h5'

best_score = 0.0


@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_lstm_layers,
            num_lstm_nodes, dropout, num_epochs):

    num_epochs = 2**num_epochs
    learning_rate = math.pow(10, learning_rate)

    print('learning rate: {0:.1e}'.format(learning_rate))
    print('num_lstm_layers: ', num_lstm_layers)
    print('num_lstm_nodes: ', num_lstm_nodes)
    print('num_epochs: ', num_epochs)
    print('dropout: ', dropout)
    print()

    model_fn = create_model_fn(learning_rate=learning_rate,
                         num_lstm_layers=num_lstm_layers,
                         num_lstm_nodes=num_lstm_nodes,
                         dropout=dropout)

    # log_dir = log_dir_name(learning_rate, num_lstm_layers,
    #                        2**num_lstm_nodes, num_epochs, dropout)

    # callback_log = TensorBoard(
    #     log_dir=log_dir,
    #     histogram_freq=1,
    #     write_graph=True,
    #     write_images=True)

    # history = model.fit(x=X_train,
    #                     y=y_train,
    #                     epochs=num_epochs,
    #                     batch_size=128,
    #                     validation_data=(X_test, y_test),
    #                     callbacks=[callback_log],
    #                     verbose=0)
    # auc = history.history['val_auc'][-1]

    pe = ImpersonalExperiment(model_fn, 'rnn', 'regression', 34, 4, 1, 60, True)
    pe.run(num_epochs, save=False)
    score = pe.get_mean_score()
    del pe
    print()
    print("Score: {0:.3}".format(score))
    print()
    global best_score

    if score > best_score:
        #model.save(path_best_model)
        best_score = score
    return score


#%%
#fitness(x=default_parameters)
#%%
search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI',  # Expected Improvement.
                            n_calls=15,
                            x0=default_parameters,
                            n_random_starts=4,
                            verbose=True)
#%%

print(search_result.fun)

sorted(zip(search_result.func_vals, search_result.x_iters))

#%%
search_result.fun

# personal / rnn / 34
# [-2, 1, 4, 6, 0.7815470221731062]
# 
# personal / rnn / 32
# [-2, 1, 6, 4, 0.46731340087392575]
# 0.5639699458628711
# impersonal / rnn / 34
#
# impersonal / rnn / 32
#
