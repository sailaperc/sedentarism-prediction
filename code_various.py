#%% 
from utils.utils_graphic import plot_heatmaps_mean, plot_heatmaps_std

plot_heatmaps_mean(users=[3, 46, -1])
plot_heatmaps_std(users=[3, 46, -1])

#%%
from preprocessing.datasets import get_lagged_dataset

df = get_lagged_dataset(nb_lags=4, period=4)
list(df.columns)[0 : len(df.columns) : int((len(df.columns)-1)/4)]

#%%
a = list(range(10))
a[0:len(a):3]
(len(df.columns)-1)/4