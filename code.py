#%%
import numpy as np
import pandas as pd
from preprocessing.studentlife_raw import get_studentlife_dataset, get_sensor_data
from preprocessing.datasets import get_lagged_dataset
from sklearn.model_selection import TimeSeriesSplit
from preprocessing.studentlife_raw import get_sensor_data
from preprocessing.model_ready import split_x_y
from utils.utils import get_user_data, get_not_user_data

#%%

def time_series_split(train_data, test_date, n_splits):
    # for dates:
    # min is the max of min dates of train and test data
    # max is the min of max dates of train and test data
    min_train = train_data.index.get_level_values(1).min()
    min_test = test_data.index.get_level_values(1).min()
    max_train = train_data.index.get_level_values(1).max()
    max_test = test_data.index.get_level_values(1).max()
    min_date = max([min_train, min_test])
    max_date = min([max_train, max_test])
    # print(f'min_train : {min_train}')
    
    # print(f'min_test : {min_test}')
    # print(f'max_train : {max_train}')
    # print(f'max_test : {max_test}')
    # print(f'min_date : {min_date}')
    # print(f'max_date : {max_date}')
    diff = max_date-min_date
    n_folds = n_splits + 1
    time_per_fold = diff / n_folds
    split_date = min_date
    for split_nb in range(n_splits):
        split_date = split_date + time_per_fold

        train_index = (train_data.index.get_level_values(1) <= split_date)
        train_data_split = train_data[train_index]


        if split_nb != n_splits-1:
            test_index_may = (test_data.index.get_level_values(1) > split_date)
            test_index_inf = (test_data.index.get_level_values(1) < (split_date+time_per_fold))
            test_index = test_index_may & test_index_inf
            test_data_split = test_data[test_index]
        else:
            print('agarro los ultimos')
            test_index = (test_data.index.get_level_values(1) > split_date)
            test_data_split = test_data[test_index]
        X_train, y_train = split_x_y(train_data_split)
        X_test, y_test = split_x_y(test_data_split)
        yield X_train, y_train, X_test, y_test

#%%
df = get_lagged_dataset()
for user in [0,1]:
    print('*' * user)
    print(user)
    print('*' * user)

    train_data = get_not_user_data(df,user)
    test_data = get_user_data(df,user)

    n_splits = 5

    for split_data in time_series_split(train_data,test_data,n_splits):
        X_train, y_train, X_test, y_test = split_data
        print(X_train.shape,y_train.shape, X_test.shape, y_test.shape)



# %%
