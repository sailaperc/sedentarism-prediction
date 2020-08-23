import numpy as np
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

from preprocessing.model_ready import get_list_of_users

from experiments.Experiment import PersonalExperiment, ImpersonalExperiment

from preprocessing.datasets import get_clean_dataset
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min

def create_rnn_model_fn(learning_rate, num_lstm_layers, num_lstm_nodes, dropout):
    def create_rnn_model():
        use_two_layers = num_lstm_layers == 2

        model = Sequential(name='rnn')

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
        return model
    return create_rnn_model

def create_mlp_model_fn(learning_rate, num_dense_nodes, dropout):
    def create_mlp_model():
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


models = {
    'personal': {
        'rnn': {
            34: (create_rnn_model_fn(-2, 1, 4, 0.7815470221731062), 2**6),
            32: (create_rnn_model_fn(-2, 1, 6, 0.46731340087392575), 2**4)
        },
        'cnn': {
            34: (create_cnn_model_fn(), 2**),
            32: (create_cnn_model_fn(), 2**)
        },
        'tcn': {
            34: (create_tcn_model_fn(), 2**),
            32: (create_tcn_model_fn(), 2**)
        },
        'mlp': {
            34: (create_mlp_model_fn(-2, 4, 0.48791799675843806), 2**6),
            32: (create_mlp_model_fn(-2, 9, 0.6489600921938838), 2**6)
        },
    },
    'impersonal': {
        'rnn': {
            34: (create_rnn_model_fn(), 2**),
            32: (create_rnn_model_fn(), 2**)
        },
        'cnn': {
            34: (create_cnn_model_fn(), 2**),
            32: (create_cnn_model_fn(), 2**)
        },
        'tcn': {
            34: (create_tcn_model_fn(), 2**),
            32: (create_tcn_model_fn(), 2**)
        },
        'mlp': {
            34: (create_mlp_model_fn(), 2**),
            32: (create_mlp_model_fn(), 2**)
        },
    }
}


def get_closests():
    '''
    Returns a dictionary with the closest centroid for each user
    '''
    df = get_clean_dataset()
    d = df.groupby(level=0)['slevel'].agg(['count', 'mean', 'std'])
    k = 2
    nb_kmean = k
    kmeans = KMeans(n_clusters=nb_kmean).fit(d)
    y = pd.DataFrame(data = {'group': kmeans.predict(d), 'user': d.reset_index().iloc[:,0].values})
    user_34_group = int(y.loc[y.user==34,'group'])
    user_32_group = int(y.loc[y.user==32,'group'])
    y.loc[y.group==user_34_group,'group'] = 34
    y.loc[y.group==user_32_group,'group'] = 32
    y = y.set_index('user').iloc[:,0].to_dict()

    return y

def run_all_experiments():
    task_type = 'regression'
    closest = get_closests()
    # model combinations
    for poi in ['personal', 'impersonal']:
        for arq in ['rnn', 'cnn', 'tcn', 'mlp']:
            need_3d_input = (arq != 'mlp')
            for user in get_list_of_users():

                closest_centroid = closest[user]  # TODO implement function
                info_model = models[poi][arq][closest_centroid]
                model = info_model[0]
                epochs = info_model[1]
                
                # dataset combinations
                for nb_lag in [1, 2, 4, 8]:
                    for period in [1, 2, 4]:
                        for gran in [30, 60]:
                            if poi == 'personal':
                                experiment = PersonalExperiment(
                                    model, arq, task_type, user, nb_lags, period, need_3d_input)
                            else:
                                experiment = ImpersonalExperiment(
                                    model, arq, task_type, user, nb_lags, period, need_3d_input)
                            experiment.run(epochs)
                            if not experiment.déjà_fait:
                                experiment.save()
                            print(experiment.get_results())
                            print(experiment.get_mean_score())
                            del experiment
