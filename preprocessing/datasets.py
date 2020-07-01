import pandas as pd
import numpy as np
from utils.utils import get_user_data, file_exists
from preprocessing.various import addSedentaryClasses
from preprocessing.various import delete_user, makeDummies, addSedentaryLevel, delete_sleep_buckets


def shift_data(df, n):
    '''
    Shift the dataset n hours. If

    :param n: number of hours to shift
    :param columns: the columns that should be shifted,
    :return:
    '''
    dfcopy = df.copy()
    columns = df.columns
    #TODO: adapt to a different granularity
    for ind, _ in dfcopy.iterrows():
        try:
            dfcopy.loc[(ind[0], ind[1]), columns] = dfcopy.loc[(ind[0], ind[1] + pd.DateOffset(hours=n)), columns]
        except KeyError:
            dfcopy.loc[(ind[0], ind[1]), columns] = np.nan
    # print(dfcopy.isna().sum())
    dfcopy.dropna(inplace=True)
    return dfcopy


def concatenate_shift_data(df, dropnan=True, nb_lags=None, period=1):
    '''
    Creates the lagged dataset calling shift_hours for every lag and then combines all the lagged datasets

    :param period: separation of lags, for example: if period = 3 and lag = 3, por a time t we will have features of t-3,
    t-6 and t-9.
    :return:
    '''
    lags = range(period * nb_lags, 0, -period)
    columns = df.columns
    n_vars = df.shape[1]
    data, names = list(), list()
    # print('Generating {0} time-lags with period equal {1} ...'.format(number_of_lags, period))
    # input sequence (t-n, ... t-1)
    for i in range(len(lags), 0, -1):
        data.append(shift_data(df, lags[i - 1]))
        names += [('{0}(t-{1})'.format(columns[j], lags[i - 1])) for j in range(n_vars)]
    data.append(df.iloc[:, -1])
    names += [('{0}'.format(columns[-1]))]

    # put it all together
    agg = pd.concat(data, axis=1)
    agg.columns = names
    # drop rows w   ith NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def generate_dataset(gran='1h', delete_inconcitencies=True, with_dummies=True, file_name=''):
    df = pd.read_pickle(f'pkl/sedentarismdata_gran{gran}.pkl')
    df.dropna(inplace=True)
    if delete_inconcitencies: 
        df = delete_user(df, 52)
    if with_dummies:
        df = makeDummies(df)
    df = addSedentaryLevel(df)
    df.to_pickle(file_name)
    return df


def get_dataset(gran='1h', delete_inconcitencies=True, with_dummies=True):
    '''
        Creates a dataset with granularity gran. It uses the preprocesed dataset  with the same granularity and makes
        some preprocessing steps (delete the user 52, make dummy variables and calculate de sLevel feature.

    '''

    file_name = f'pkl/datasets/dataset_gran{gran}.pkl'
    if not file_exists(file_name):
        print('Dataset does not exist.')
        print(f'Generating dataset with gran: {gran}')
        generate_dataset(gran, delete_inconcitencies, with_dummies, file_name)

    return pd.read_pickle(file_name)


def generate_lagged_dataset(file_name, model_type, included_data, nb_lags=1, period=1, gran='1h'):
    '''
    Calls series_to_supervised for every user (otherwise user information would be merged) and then combines it.
    The resulting dataset is saved in the path 'pkl/datasets/gran{}_period{}_lags{}.pkl'

    :param gran: granularity. e.g. '1h', '30m', '2h', etc
    '''

    assert (model_type == 'regression' or model_type == 'classification'), 'Not a valid model type.'
    assert (included_data == 'ws' or included_data == 'wos'), f'included_data must be ws or wos'

    df = get_dataset(gran=gran)

    if included_data == 'wos':
        df = delete_sleep_buckets(df)
    if model_type == 'classification':
        df = addSedentaryClasses(df)

    data = list()
    for i in df.index.get_level_values(0).drop_duplicates():
        d = concatenate_shift_data(get_user_data(df, i), nb_lags=nb_lags, period=period)
        data.append(d)
    df = pd.concat(data, axis=0)
    df.to_pickle(file_name)
    del df


def get_lagged_dataset(model_type, included_data, user=-1, nb_lags=1, period=1, gran='1h'):
    '''
    Get a specific and already generated dataset based on nb_lags, period, gran.
    If personal is true, only returns the specific users data

    '''

    filename = f'pkl/lagged_datasets/{model_type}_{included_data}_gran{gran}_period{period}_lags{nb_lags}.pkl'
    if not file_exists(filename):
        print('Lagged dataset not found.')
        print(
            f'Generating lagged dataset with period '
            f'gran: {gran}, period: {period} and nb_lags={nb_lags} for a {model_type} model (with data {included_data}.)')
        generate_lagged_dataset(filename, model_type, included_data, nb_lags, period, gran)
    data = pd.read_pickle(filename)
    if user != -1:
        return get_user_data(data, user)
    else:
        return data



