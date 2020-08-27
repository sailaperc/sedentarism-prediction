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

dim_learning_rate = Integer(low=-6, high=-2,name='learning_rate')
dim_num_dense_nodes = Integer(low=2, high=9, name='num_dense_nodes')
dim_dropout = Real(low=.0, high=.8, name='dropout')
dim_num_epochs = Integer(low=2, high=6, name='num_epochs')
dim_batch_size = Integer(low=3, high=8, name='batch_size')

dimensions = [dim_learning_rate,
              dim_num_dense_nodes,
              dim_dropout,
              dim_num_epochs,
              dim_batch_size]
default_parameters = [-5, 4,.3, 2 ,4]

def create_model(learning_rate, num_dense_nodes, dropout):

    def create_model():
        model = Sequential(name='mlp')

        for i in range(num_dense_nodes,1,-1):
            model.add(Dense(2**i, activation='relu'))
            model.add(Dropout(dropout))
        model.add(Dense(1, activation='linear'))
        optimizer = Adam(lr=learning_rate)
        model.compile(optimizer=optimizer,
                    loss='MSE',
                    metrics=keras.metrics.MSE)
        return model
    return create_model
    

best_score = 0.0

@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_nodes, dropout, num_epochs, batch_size):
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('dropout:', dropout)
    print('num_dense_nodes:', num_dense_nodes)
    print('num_epochs: ', num_epochs)
    print('batch_size: ', batch_size)
    print()
    num_epochs = 2**num_epochs
    batch_size = 2**batch_size
    learning_rate = math.pow(10,learning_rate)
    
    model_fn = create_model(learning_rate=learning_rate,
                         num_dense_nodes=num_dense_nodes,
                         dropout=dropout
                         )

    pe = PersonalExperiment(model_fn, 'mlp', 'regression', 32, 4, 1, 60, False)
    pe.run(num_epochs, batch_size)
    score = pe.get_mean_score()
    del pe
    print()
    print("Score: {0:.3}".format(score))
    print()
    global best_score

    if score > best_score:
        best_score = score
    return score


# %%
#fitness(x=default_parameters)

#%%
search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI',  # Expected Improvement.
                            n_calls=40,
                            verbose=True)


print(search_result.fun)

sorted(zip(search_result.func_vals, search_result.x_iters))

#%%
# personal / mlp / 34
# (0.47701561717439234, [-2, 9, 0.2839842121499357, 4, 3]

# personal / mlp / 32
# (0.6090751778316554, [-3, 7, 0.29014804584108833, 6, 3]),

# impersonal / mlp / 34
# 