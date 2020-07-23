#%%
import numpy as np
import pandas as pd
from preprocessing.studentlife_raw import get_studentlife_dataset, get_sensor_data
from preprocessing.datasets import get_lagged_dataset
from sklearn.model_selection import TimeSeriesSplit
from preprocessing.studentlife_raw import get_sensor_data

#%%
df = get_lagged_dataset()
df
# %%
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4], [1,3],[3,2],[4,1],[3,3]])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
tscv = TimeSeriesSplit(n_splits=3)
print(tscv)
for train, test in tscv.split(X):
    print("%s %s" % (train, test))

# %%
test_size = 10 // 4
n_samples = 10
n_folds = 4
test_starts = range(test_size + n_samples % n_folds,
    n_samples, test_size)
for ts in test_starts:
    print(ts)
    print(f'train: {y[:ts]}')
    print(f'test {y[ts:ts+test_size]}')
    print('*' * 8)