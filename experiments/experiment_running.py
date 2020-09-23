import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import Conv1D, Dense, Flatten, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout
from utils.utils import get_granularity_from_minutes
from tcn import TCN
from utils.utils import file_exists

import math

from skopt import load

import time

from preprocessing.model_ready import get_list_of_users

from experiments.Experiment import PersonalExperiment, ImpersonalExperiment

from preprocessing.datasets import get_clean_dataset
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min


def create_cnn_model_fn(num_filters, num_kernels, conv_dropout, num_dense_nodes, dense_dropout):
    def create_model():
        model = Sequential(name='cnn')
        model.add(Conv1D(filters=2**num_filters,
                         kernel_size=int(num_kernels),
                         activation='relu',
                         padding='causal'))
        model.add(Dropout(conv_dropout))
        model.add(Flatten())
        if num_dense_nodes > 0:
            model.add(Dense(2**num_dense_nodes, activation='relu'))
            model.add(Dropout(dense_dropout))
        model.add(Dense(1, activation='linear'))

        model.compile(loss='MSE',
                      optimizer='adam', metrics=keras.metrics.MSE
                      )
        return model
    return create_model


def create_mlp_model_fn(num_dense_nodes, num_dense_layers, use_batch_norm, dropout):
    def create_model():
        model = Sequential(name='mlp')
        for _ in range(num_dense_layers):
            model.add(Dense(2**num_dense_nodes, activation='relu'))
            if use_batch_norm == 1:
                model.add(BatchNormalization())
            model.add(Dropout(dropout))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam',
                      loss='MSE',
                      metrics=keras.metrics.MSE)
        return model
    return create_model


def create_rnn_model_fn(num_lstm_layers, num_lstm_nodes, lstm_dropout, num_dense_nodes, dense_dropout):
    def create_model():
        use_two_layers = num_lstm_layers == 2

        model = Sequential(name='rnn')
        model.add(LSTM(2**num_lstm_nodes, return_sequences=use_two_layers))
        model.add(Dropout(lstm_dropout))

        if use_two_layers:
            model.add(LSTM(2**(num_lstm_nodes-1), return_sequences=False))
            model.add(Dropout(lstm_dropout))

        if num_dense_nodes > 0:
            model.add(Dense(2**num_dense_nodes, activation='relu'))
            model.add(Dropout(dense_dropout))

        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam',
                      loss='MSE', metrics=keras.metrics.MSE)
        return model
    return create_model


def create_tcn_model_fn(num_filters, kernel_size, dropout, use_skip_connections, use_batch_norm, nb_lags):
    nb_dilations = max(
        math.ceil(math.log(nb_lags/kernel_size, kernel_size)+1), 2)
    if use_skip_connections == 1 and nb_dilations == 1:
        nb_dilations += 1
    print('dilations:', nb_dilations)
    print('receptive field: ', (kernel_size**(nb_dilations-1))*kernel_size)

    def create_model():
        model = Sequential(name='tcn')
        dilations = [2 ** i for i in range(nb_dilations)]
        model.add(TCN(
            nb_filters=2**num_filters,
            kernel_size=int(kernel_size),
            nb_stacks=1,
            dilations=dilations,
            padding='causal',
            use_skip_connections=use_skip_connections,
            dropout_rate=dropout,
            return_sequences=False,
            activation='relu',
            kernel_initializer='he_normal',
            use_batch_norm=use_batch_norm,
            use_layer_norm=False))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='MSE',
                      optimizer='adam', metrics=keras.metrics.MSE
                      )
        return model
    return create_model


def get_closests():
    '''
    Returns a dictionary with the closest centroid for each user
    '''
    df = get_clean_dataset()
    d = df.groupby(level=0)['slevel'].agg(['count', 'mean', 'std'])
    k = 2
    nb_kmean = k
    kmeans = KMeans(n_clusters=nb_kmean).fit(d)
    y = pd.DataFrame(data={'group': kmeans.predict(
        d), 'user': d.reset_index().iloc[:, 0].values})
    user_34_group = int(y.loc[y.user == 34, 'group'])
    user_32_group = int(y.loc[y.user == 32, 'group'])
    y.loc[y.group == user_34_group, 'group'] = 34
    y.loc[y.group == user_32_group, 'group'] = 32
    y = y.set_index('user').iloc[:, 0].to_dict()

    return y


def get_model(arch, *args,):
    model_fn_dict = {'mlp': create_mlp_model_fn,
                     'cnn': create_cnn_model_fn,
                     'rnn': create_rnn_model_fn,
                     'tcn': create_tcn_model_fn}
    return model_fn_dict[arch](*args)


def get_model_info(arch, centroid, model_type):
    checkpoint_file = f'pkl/tunning/checkpoint_{arch}_{centroid}_{model_type}.pkl'
    res = load(checkpoint_file)
    return sorted(zip(res.func_vals, res.x_iters))[0][1]


def run_all_experiments(verbose=0):
    task_type = 'regression'
    closest = get_closests()
    users = get_list_of_users()
    cant_experiments = 2 * 4 * 4 * 3 * 2 * len(users)
    c = 0
    # model combinations
    for poi in ['per', 'imp']:
        for arch in ['rnn', 'cnn', 'tcn', 'mlp']:
            for user in users:
                # dataset combinations
                for nb_lags in [1, 2, 4, 8]:
                    for period in [1, 2, 4]:
                        for gran in [30, 60]:
                            name = f'_regression_gran{get_granularity_from_minutes(gran)}_period{period}_lags{nb_lags}_model-{arch}_user{user}_{poi}'
                            file_name = f'./pkl/experiments/{name}.pkl'
                            if not file_exists(file_name):
                                need_3d_input = (arch != 'mlp')
                                closest_centroid = closest[user]
                                model_info = get_model_info(
                                    arch, closest_centroid, poi)
                                [nb_epochs, batch_size] = model_info[-2:]
                                if arch != 'tcn':
                                    model = get_model(arch, *model_info[:-2])
                                else:
                                    model = get_model(
                                        arch, *(model_info[:-2]+[nb_lags]))
                                if poi == 'per':
                                    experiment = PersonalExperiment(
                                        model, arch, task_type, user, nb_lags, period, gran, need_3d_input)
                                else:
                                    experiment = ImpersonalExperiment(
                                        model, arch, task_type, user, nb_lags, period, gran, need_3d_input)
                                experiment.run(2**nb_epochs, 2 **
                                               batch_size, verbose=verbose)
                                if not experiment.déjà_fait:
                                    experiment.save()
                                # print(experiment.get_results())
                                # print(experiment.get_mean_score())
                                del experiment
                            c += 1
                            print('#' * 4)
                            print(
                                f'{c}/{cant_experiments} fishished experiments')
                            print('#' * 4)
