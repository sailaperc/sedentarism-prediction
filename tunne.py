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
from tensorflow.keras.layers import Reshape, MaxPooling1D
from tensorflow.keras.layers import Conv1D, Dense, Flatten, LSTM
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout

from tcn import TCN

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


dim_num_filters = Integer(low=2, high=8, name='num_filters')
dim_num_kernels = Integer(low=1, high=4, name='num_kernels')
dim_conv_dropout = Real(low=.0, high=.8, name='conv_dropout')
dim_num_dense_nodes = Integer(low=0, high=6, name='num_dense_nodes')
dim_dense_dropout = Real(low=.0, high=.8, name='dense_dropout')
dim_num_epochs = Integer(low=2, high=6, name='num_epochs')
dim_batch_size = Integer(low=3, high=8, name='batch_size')

dimensions = [dim_learning_rate,
              dim_num_filters,
              dim_num_kernels,
              dim_conv_dropout,
              dim_num_dense_nodes,
              dim_dense_dropout,
              dim_num_epochs,
              dim_batch_size]

default_parameters = [-3, 4, 2, .5, 4, .5, 2, 4]


def create_model_fn(learning_rate, num_filters, num_kernels, conv_dropout, num_dense_nodes, dense_dropout):
    def create_model():
        model = Sequential(name='cnn')
        model.add(TCN(
            nb_filters=64,
            kernel_size=2,
            nb_stacks=1,
            dilations=(1, 2, 4, 8, 16, 32),
            padding='causal',
            use_skip_connections=False,
            dropout_rate=0.0,
            return_sequences=False,
            activation='relu',
            kernel_initializer='he_normal',
            use_batch_norm=False,
            use_layer_norm=False))

        model.compile(loss='MSE',
                      optimizer=optimizer,
                      )
        return model
    return create_model

best_score = 0.0

@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_filters, num_kernels, conv_dropout, num_dense_nodes, dense_dropout, num_epochs, batch_size):

    learning_rate = math.pow(10,learning_rate)

    print('learning rate: {0:.1e}'.format(learning_rate))
    print('num_filters:', num_filters)
    print('num_kernels:', num_kernels)
    print('conv_dropout:', conv_dropout)
    print('num_dense_nodes:', num_dense_nodes)
    print('dense_dropout:', dense_dropout)
    print('num_epochs:', num_epochs)
    print('batch_size: ', batch_size)
    print()

    
    model_fn = create_model_fn(learning_rate=learning_rate, num_filters=num_filters, num_kernels=num_kernels, conv_dropout=conv_dropout, num_dense_nodes=num_dense_nodes, dense_dropout=dense_dropout)

    pe = PersonalExperiment(model_fn, 'cnn', 'regression', 32, 4, 1, 60, True)
    pe.run(2**num_epochs, 2**batch_size)
    score = pe.get_mean_score()
    del pe
    print()
    print("Score: {0:.3}".format(score))
    print()
    global best_score

    if score > best_score:
        best_score = score
    return score


#%%
#fitness(x=default_parameters)

#%%
search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI',  # Expected Improvement.
                            n_calls=40,
                            verbose=True)

#%%

print(search_result.fun)

sorted(zip(search_result.func_vals, search_result.x_iters))

#%%
# personal / cnn / 34
# [(0.46322976129129784, [-2, 7, 2, 0.5699083979399923, 6, 0.10119762716042291, 5, 3])

# personal / cnn / 32   
# (0.6199441317952524, [-2, 8, 1, 0.0, 4, 0.8, 6, 6])

# impersonal / cnn / 34

#%%

model = compiled_tcn(return_sequences=False,
    num_feat=number_of_features,
    num_classes=0,
    nb_filters=8,
    kernel_size=1,
    dilations=[1, 2, 4],
    # dilations=[2 ** i for i in range(2, 5)],
    nb_stacks=1,
    max_len=time_lags,
    use_skip_connections=True,
    regression=True,
    dropout_rate=0.6)
