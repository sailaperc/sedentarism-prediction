from os import listdir
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import random 
import pandas as pd
import pickle
import time
import os 
from matplotlib.ticker import MaxNLocator
from matplotlib import lines
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import seaborn as sns
from preprocessing.datasets import get_clean_dataset
from utils.utils import get_granularity_from_minutes, get_list_of_users, get_experiment_combinations
from experiments.experiment_running import get_closests

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

def get_experiments_data(only_gpu, with_y_test_pred=False):
    df = pd.read_pickle(f'../pkl/{get_experiments_directory(only_gpu)}/experiments_df.pkl')
    if not with_y_test_pred:
        return df.loc[:, [col for col in df.columns if col!='y_test_pred']]
    return df

def get_experiments_directory(only_gpu):
    return 'experiments-gpu' if only_gpu==True else 'experiments'

def generate_df_from_experiments(only_gpu = True):

    start = time.time()
    rows = []
    combs = get_experiment_combinations()
    closest = get_closests()
    for poi, arch, user, gran, nb_lags, period in combs:
        name = f'_regression_gran{get_granularity_from_minutes(gran)}_period{period}_lags{nb_lags}_model-{arch}_user{user}_{poi}'
        print(name)
        experiments_directory = get_experiments_directory(only_gpu)
        filename = f'../pkl/{experiments_directory}/{name}.pkl'
        exp_data = pkl.load(open(filename, 'rb'))
        centroid = closest[user]
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
            'time_to_train': exp_data['time_to_train'],
            'centroid': 'low_met' if centroid == 34 else 'high_met'
        }
        rows.append(new_row)

    df = pd.DataFrame(rows)
    df['mean_score'] = df.scores.apply(lambda x : np.mean(x))
    df['mean_time'] = df.time_to_train.apply(lambda x : np.mean(x))

    df[[f'score_{i}' for i in range(5)]] = pd.DataFrame(df.scores.tolist())
    del df['scores']

    df[[f'time_{i}' for i in range(5)]] = pd.DataFrame(df.time_to_train.tolist()) 
    del df['time_to_train']


    filename = f'../pkl/{experiments_directory}/experiments_df.pkl'
    df.to_pickle(filename)
    print(f'This took {round((time.time() - start)/60, 3)}')

def rank_results(only_gpu = False, comp_col='arch', rank_by='score', based_on='user', ix=-1, **kwargs):
    '''
    This function generates a table that ranks the specify comp_col columns
    based on its performance (mean_score col) for all the users
    
    '''
    # TODO implement from agregation function
    df = get_experiments_data(only_gpu)

    assert comp_col in df.columns, f'comp_col must be one of {df.columns}'
    assert comp_col not in kwargs.keys() , f'comp_col cant be a filter keyword'
    assert all(k in df.columns for k in kwargs.keys()) , f'kwargs must be one of {df.columns}'
    
    col_values = list(df[comp_col].drop_duplicates())
    nb_values = len(col_values)
    rank_col_names = [f'Puesto {i}' for i in range(1,nb_values+1)]
    
    if rank_by in ['score','time']:
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

def filter_exp(only_gpu=False, **kwargs):
    df = get_experiments_data(only_gpu)
    for k,v in kwargs.items():
        df = df.loc[df[k]==v]
    return df

def order_exp_by(only_gpu=False, ix=-1, rank_by='score', **kwargs):
    df = filter_exp(only_gpu, **kwargs)
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
    if return_shapes: 
        return y_test, y_pred, shapes
    else: 
        return y_test, y_pred

def print_results(fromi=1, toi=5, archs=['rnn', 'tcn', 'cnn', 'mlp'], poi='per', user=32, lags=1, period=1, gran=60):
    df = get_experiments_data(False, with_y_test_pred=True)
    exp = df.loc[((df.poi==poi) & (df.user==user) & (df.nb_lags==lags) & (df.period==period) & (df.gran==gran))]
    plt.close()
    width = 4 + 2*(toi-fromi+1)
    plt.figure(figsize=(width,4))
    first_pass = True
    for arch in archs:
        exp_arch = exp.loc[df.arch==arch,:]
        exp_data = exp_arch.y_test_pred.values[0][fromi-1:toi]
        y_test, y_pred, shapes = get_test_predicted_arrays(exp_data, return_shapes=True)
        lw = .6
        if first_pass:
            plt.plot(y_test, label='Test', lw=lw)
            first_pass = False
            acc_shapes = np.cumsum(np.array(shapes))
            for shape in acc_shapes:
                plt.axvline(shape, color='black', ls='--')
            plt.xlim(0, acc_shapes[-1])
        plt.plot(y_pred, label=f'Predicho ({arch.upper()})', lw=lw)
    plt.axhline(1.5, color='red',ls=':')
    plt.ylim(0,8.5)
    plt.tick_params(
        axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.ylabel('MET')
    plt.tight_layout()
    plt.legend()
    plt.show()

def plot_iterations_time_pattern(max_minutes=None,**kwargs):
    df = filter_exp(only_gpu=True, **kwargs)
    column_names = [f'time_{i}' for i in range(5)]
    plt.close()
    ax = plt.figure(figsize=(6, 6)).gca()

    archs_colors={'rnn': 'b', 'tcn': 'g', 'cnn': 'r', 'mlp': 'm'}
    its = np.arange(1,6)
    first_pass = True
    for arch in archs_colors.keys():
        exp_arch = df.loc[df.arch==arch,:]
        
        to_plot = exp_arch.loc[:, column_names].values.tolist()
        for list in to_plot:
            ax.plot(its, list, lw=.01, color=archs_colors[arch])
            first_pass = False
        first_pass = True

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(1,5)
    if max_minutes != None: 
        ax.set_ylim(0, max_minutes)
    plt.ylabel('Tiempo')
    plt.xlabel('Iteración')

    handles = [lines.Line2D([], [], color=c,
                            markersize=15, label=k.upper()) for k,c in archs_colors.items()]
    plt.legend(loc='upper left', handles=handles)

    plt.show()

def plot_iterations_score_pattern(max_mse=None, **kwargs):
    df = filter_exp(only_gpu=False, **kwargs)
    column_names = [f'score_{i}' for i in range(5)]
    plt.close()
    ax = plt.figure(figsize=(6, 6)).gca()

    archs_colors={'rnn': 'b', 'tcn': 'g', 'cnn': 'r', 'mlp': 'm'}
    its = np.arange(1,6)
    first_pass = True
    for arch in archs_colors.keys():
        exp_arch = df.loc[df.arch==arch,:]
        
        to_plot = exp_arch.loc[:, column_names].values.tolist()
        for list in to_plot:
            ax.plot(its, list, lw=.01, color=archs_colors[arch])
            first_pass = False
        first_pass = True

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(1,5)
    if max_mse != None: 
        ax.set_ylim(0, max_mse)
    plt.ylabel('MSE')
    plt.xlabel('Iteración')

    handles = [lines.Line2D([], [], color=c,
                            markersize=15, label=k.upper()) for k,c in archs_colors.items()]
    plt.legend(loc='upper left', handles=handles)

    plt.show()



def plot_clusters_performance(ix=-1, **kwargs):
    df = get_clean_dataset()
    d = df.groupby(level=0)['slevel'].agg(['count', 'mean', 'std'])
    nb_kmean = 2
    kmeans = KMeans(n_clusters=nb_kmean).fit(d)


    def get_mse_and_best_arch(x):
        return x.sort_values(by='mean_score').loc[:,['mean_score','arch']].iloc[0,:]

    df_exp = order_exp_by(only_gpu=False, ix=ix, **kwargs)
    per_user_best = df_exp.groupby('user').apply(get_mse_and_best_arch) 
    per_user_best.arch = per_user_best.arch.str.upper()
    d = pd.concat([d, per_user_best], axis=1)
    d.columns = ['Cantidad buckets', 'Promedio MET','Desviacion Estándar MET', 'MSE', 'Arquitectura']

    colors={'RNN': 'b', 'TCN': 'g', 'CNN': 'r', 'MLP': 'm'}

    g = sns.relplot(x='Cantidad buckets',
                    y='Promedio MET',
                    hue='Arquitectura',
                    size='MSE',
                    sizes=(100, 500),
                    alpha=.6,
                    data=d,
                    palette=colors)

    g.ax.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='r', marker='x')

    plt.show()
