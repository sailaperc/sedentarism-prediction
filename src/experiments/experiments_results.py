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
    df = pd.read_pickle('../pkl/experiments/experiments_df.pkl')
    if not with_y_test_pred:
        return df.loc[:, [col for col in df.columns if col!='y_test_pred']]
    return df

def generate_df_from_experiments():

    start = time.time()
    rows = []
    combs = get_experiment_combinations()
    for poi, arch, user, gran, nb_lags, period in combs:
        name = f'_regression_gran{get_granularity_from_minutes(gran)}_period{period}_lags{nb_lags}_model-{arch}_user{user}_{poi}'
        filename = f'../pkl/experiments/{name}.pkl'
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


    filename = '../pkl/experiments/experiments_df.pkl'
    df.to_pickle(filename)
    print(f'This took {round((time.time() - start)/60, 3)}')

def rank_results(comp_col='arch', rank_by='score', based_on='user', ix=-1, **kwargs):
    '''
    This function generates a table that ranks the specify comp_col columns
    based on its performance (mean_score col) for all the users
    
    '''
    # TODO implement from agregation function
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

def check_results_correctness():
    # este codigo compara los datos de y_test de los experimentos y 
    # los que estan en el dataset, para ver si concuerdan
    # en algunos hay una diferencia de uno, pero nada mas
    poi = 'per'
    arch = 'mlp'
    nb_lags = 4
    period = 4
    gran = 60
    user = 32
    df = get_experiments_data(with_y_test_pred=True)
    exp = df.loc[((df.poi==poi) & (df.arch==arch) & (df.nb_lags==nb_lags) & (df.period == period) & (df.user==user) & (df.gran==gran))]
    exp = exp.y_test_pred.values[0]
    y_test, y_pred,shapes = get_test_predicted_arrays(exp, return_shapes=True)
    print(y_test.shape, y_pred.shape)
    dataset = get_lagged_dataset(user=user, nb_lags=nb_lags, period=period, nb_min=gran)
    y_test_total = dataset.slevel.values
    y_test_exp = y_test
    nb_cases = y_test_total.shape[0]
    nb_exp_cases = y_test_exp.shape[0]
    diff = nb_cases - nb_exp_cases
    print(f'total nb of cases: {nb_cases}')
    print(f'total nb of cases in the exp: {nb_exp_cases}')
    print(f'diff: {diff}')
    y_test_total_cut = y_test_total[diff:]
    print(y_test_total_cut.shape)
    print(y_test_exp.shape)
    new_df = pd.DataFrame(data={'total': y_test_total_cut, 'exp': y_test_exp})
    shapes = list(np.cumsum(shapes))
    shapes = [0] + shapes[:-1]
    print(shapes)
    for i in range(len(shapes)):
        arr1 = y_test_total_cut[shapes[i]:shapes[i]+10]
        arr2 = exp[i][0][:10]
        new_df = pd.DataFrame(data={'total': arr1, 'exp': arr2})
        print(new_df)        

def get_test_predicted_arrays(exp_data, return_shapes=False):
    zipped = zip(*exp_data)
    l = list(zipped)
    l[1] = [np.squeeze(a) for a in l[1]]
    y_test = np.concatenate(l[0]) 
    y_pred = np.concatenate(l[1])
    shapes = [arr.shape[0] for arr in l[0]]
    print(shapes) 
    return y_test, y_pred, shapes

def print_results(fromi=1, toi=5, archs=['rnn', 'tcn', 'cnn', 'mlp'], poi='per', user=32, lags=1, period=1, gran=60):
    df = get_experiments_data(with_y_test_pred=True)
    exp = df.loc[((df.poi==poi) & (df.user==user) & (df.nb_lags==lags) & (df.period==period) & (df.gran==gran))]
    plt.close()
    width = 2 + 2*(toi-fromi+1)
    plt.figure(figsize=(width,4))
    print(width)
    first_pass = True
    for arch in archs:
        exp_arch = exp.loc[df.arch==arch,:]
        exp_data = exp_arch.y_test_pred.values[0][fromi-1:toi]
        y_test, y_pred = get_test_predicted_arrays(exp_data)
        lw = .6
        if first_pass:
            plt.plot(y_test, label='Test', lw=lw)
            first_pass = False
        plt.plot(y_pred, label=f'Predicho ({arch})', lw=lw)
    plt.legend()
    plt.show()