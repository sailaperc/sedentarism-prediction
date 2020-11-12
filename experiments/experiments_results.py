from os import listdir
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import random 
import pandas as pd
import pickle
from utils.utils import get_granularity_from_minutes
from utils.utils import get_list_of_users 
import time
import os 
from utils.utils import get_experiment_combinations

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
    rows = []
    combs = get_experiment_combinations()
    for poi, arch, user, gran, nb_lags, period in combs:
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

    df[[f'score_{i}' for i in range(5)]] = pd.DataFrame(df.scores.tolist())
    del df['scores']

    df[[f'time_{i}' for i in range(5)]] = pd.DataFrame(df.time_to_train.tolist()) 
    del df['time_to_train']


    filename = './pkl/experiments/experiments_df.pkl'
    df.to_pickle(filename)
    print(f'This took {round((time.time() - start)/60, 3)}')

def rank_results(comp_col='arch', rank_by='score', based_on='user', ix=-1, **kwargs):
    '''
    This function generates a table that ranks the specify comp_col columns
    based on its performance (mean_score col) for all the users
    
    '''
    df = get_experiments_data()

    assert comp_col in df.columns, f'comp_col must be one of {df.columns}'
    assert comp_col not in kwargs.keys() , f'comp_col cant be a filter keyword'
    assert all(k in df.columns for k in kwargs.keys()) , f'kwargs must be one of {df.columns}'
    
    col_values = list(df[comp_col].drop_duplicates())
    nb_values = len(col_values)
    rank_col_names = [f'rank{i}' for i in range(1,nb_values+1)]
    
    if ix>=0:
        rank_by = f'{rank_by}_{ix}'
    else: rank_by = f'mean_{rank_by}'
        

    for k,v in kwargs.items():
        df = df.loc[df[k]==v]
    
    rows = []
    for bo in df[based_on].drop_duplicates():
        sorted_scores = df.loc[(df[based_on]==bo)][[comp_col,rank_by]].sort_values(rank_by, ascending=True).drop_duplicates(subset=[comp_col])
        best_based_on = sorted_scores.iloc[0:nb_values,0].values
        best_rank_by = np.round(sorted_scores.iloc[0:nb_values,1].values,4)

        row = {'bo': bo, 'best_based_on': best_based_on, 'best_rank_by': best_rank_by }
        rows.append(row)

    results = pd.DataFrame(rows)
    results[rank_col_names] = pd.DataFrame(results.best_based_on.tolist(), index=results.index)
    del results['best_based_on']

    summarize = pd.DataFrame(columns=results.columns, index=col_values)
    for i in summarize.columns:
        for j in summarize.index.values:
            summarize.at[j,i] = sum(results[i]==j)
    del summarize['bo']
    del summarize['best_rank_by']
    return summarize, results

def filter_exp(rank_by='score', ix=-1, **kwargs):
    df = get_experiments_data()
    for k,v in kwargs.items():
        df = df.loc[df[k]==v]
    if ix>=0:
        rank_by = f'{rank_by}_{ix}'
    else: rank_by = f'mean_{rank_by}'
    return df.sort_values(by=rank_by)