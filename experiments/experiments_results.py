from os import listdir
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import random 
import pandas as pd
import pickle
from utils.utils import get_granularity_from_minutes
from preprocessing.model_ready import get_list_of_users
import time
import os 

def get_classification_results(keywords):
    return [(f[0:-4], pkl.load(open(f'./pkl/results/{f}', 'rb'))) for f in listdir('./pkl/results') if all(f'_{k}' in f for k in keywords)]


def print_classification_results(keywords):
    (names, results) = map(list, zip(*get_classification_results(keywords)))
    show_metric('', 'RMSE', names, results)

def show_metric(title, ylabel, labels, data):
    user_labels = get_list_of_users()
    users_range = np.arange(1, len(user_labels))

    plt.close()
    for d in data:
        plt.scatter(users_range, d, marker='s', c=(random.random(), random.random(), random.random()))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('User')
    plt.legend(labels,
               loc='best')
    plt.xticks(users_range, user_labels, rotation='vertical')
    plt.grid(True)
    plt.show()

def get_experiments_data(with_y_test_pred=False):
    df = pd.read_pickle('./pkl/experiments/experiments_df.pkl')
    if not with_y_test_pred:
        return df.loc[:, [col for col in df.columns if col!='y_test_pred']]
    return df


def generate_df_from_experiments():
    start = time.time()
    users = get_list_of_users()
    rows = []
    for poi in ['per']:
        for arch in ['rnn', 'cnn', 'tcn', 'mlp']:
            for user in users:
                for gran in [30, 60]:
                    for nb_lags in [1, 2, 4, 8]:
                        for period in [1, 2, 4]:
                            name = f'_regression_gran{get_granularity_from_minutes(gran)}_period{period}_lags{nb_lags}_model-{arch}_user{user}_{poi}'
                            filename = f'./pkl/experiments/{name}.pkl'
                            #print(poi,arch,user,gran,nb_lags,period)
                            exp_data = pkl.load(open(filename, 'rb'))
                            #print(exp_data.keys())
                            new_row = {
                                'poi': poi,
                                'arch': arch,
                                'user': user,
                                'gran': gran,
                                'nb_lags': nb_lags,
                                'period': period,
                                'scores': exp_data['scores'],
                                'nb_params': exp_data['nb_params'],
                                'y_test_pred': exp_data['y_test_pred'],
                                'time_to_train': exp_data['time_to_train']
                            }
                            rows.append(new_row)
    df = pd.DataFrame(rows)
    df['mean_score'] = df.scores.apply(lambda x : np.mean(x))
    df['mean_time'] = df.time_to_train.apply(lambda x : np.mean(x))
    filename = './pkl/experiments/experiments_df.pkl'
    os.remove(filename)
    df.to_pickle(filename)
    print(f'This took {round((time.time() - start)/60, 3)}')