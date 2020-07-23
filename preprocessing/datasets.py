import pandas as pd
import numpy as np
from utils.utils import get_user_data, file_exists
from preprocessing.various import delete_user, makeDummies, addSedentaryLevel, delete_sleep_buckets, addSedentaryClasses, downgrade_datatypes

def shift_data(df,  nb_lags, period, model_type, dropnan=False):
    '''
    Creates the lagged dataset calling shift_hours for every lag and then combines all the lagged datasets

    :param period: separation of lags, for example: if period = 3 and lag = 3, por a time t we will have features of t-3,
    t-6 and t-9.
    :return:
    '''
    target = 'slevel'
    if model_type == 'classification':
        target = 'sclass'
    # this range goes backwards so older lags are append at first
    lags = range(period * nb_lags, 0, -period)
    # print('Generating {0} time-lags with period equal {1} ...'.format(number_of_lags, period))
    # input sequence (t-n, ... t-1)
    result = pd.DataFrame(index=df.index)
    for i in lags:
        to_shift = df.shift(i)
        result = result.join(to_shift, rsuffix=f'(t-{i})')
    result.columns = [f'{col}(t-{lags[0]})' if not col.endswith(')') else col for col in result.columns]
    result = result.join(df.loc[:,target])
    
    # drop rows with NaN values
    if dropnan:
        result.dropna(inplace=True)
    return result


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

    list_dfs = [shift_data(get_user_data(df,i),nb_lags,period, model_type) \
        for i in df.index.get_level_values(0).drop_duplicates()]

    result = pd.concat(list_dfs, axis=0)
    downgrade_datatypes(result).to_pickle(file_name)
    del result


def get_lagged_dataset(model_type='regression', included_data='ws', user=-1, nb_lags=1, period=1, gran='1h'):
    '''
    Get a specific and already generated dataset based on nb_lags, period, gran.
    If personal is true, only returns the specific users data

    '''

    filename = f'pkl/lagged_datasets/{model_type}_{included_data}_gran{gran}_period{period}_lags{nb_lags}.pkl'
    if not file_exists(filename):
        print('Lagged dataset not found.')
        print(
            f'Generating lagged dataset with period '
            f'gran: {gran}, period: {period} and nb_lags:{nb_lags} for a {model_type} model (with data {included_data}.)')
        generate_lagged_dataset(filename, model_type, included_data, nb_lags, period, gran)
    data = downgrade_datatypes(pd.read_pickle(filename))
    if user != -1:
        return get_user_data(data, user)
    return data


def generate_dataset(gran, file_name, dropna, delete_inconcitencies, with_dummies, from_disc):
    df = pd.read_pickle(f'pkl/sedentarismdata_gran{gran}.pkl')
    if dropna: 
        df.dropna(inplace=True)
    if delete_inconcitencies: 
        df = delete_user(df, 52)
    if with_dummies:
        df = makeDummies(df)
    df = addSedentaryLevel(df)
    if from_disc:
        downgrade_datatypes(df).to_pickle(file_name)
    return df


def get_dataset(gran='1h', dropna=True, delete_inconcitencies=True, with_dummies=True, from_disc=True):
    '''
        Creates a dataset with granularity gran. It uses the preprocesed dataset with the same granularity and makes
        some preprocessing steps (delete the user 52, make dummy variables, drops nans rows and calculate de sLevel feature).


        from_disc: if True will return the dataset located in storage. If there is no one,
        it will create and save it. If False, it will create and return the dataset with 
        the specified options, without saving it.
    '''

    file_name = f'pkl/datasets/dataset_gran{gran}.pkl'
    if not from_disc:
        print('Creating dataset on the fly.')
        return generate_dataset(gran, file_name, dropna, delete_inconcitencies, with_dummies, from_disc)
    elif not file_exists(file_name) and from_disc:
        print('Dataset does not exist.')
        print(f'Generating dataset with gran: {gran}')
        generate_dataset(gran, file_name, dropna, delete_inconcitencies, with_dummies, from_disc)
    return downgrade_datatypes(pd.read_pickle(file_name))


