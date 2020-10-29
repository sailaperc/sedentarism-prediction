#%% 
from utils.utils_graphic import plot_heatmaps_mean, plot_heatmaps_std

plot_heatmaps_mean(users=[3, 46, -1])
plot_heatmaps_std(users=[3, 46, -1])

#%%
from preprocessing.datasets import get_lagged_dataset
get_lagged_dataset(nb_lags=2,period=1, nb_min=60).shape
#%%
a = list(range(10))
a[0:len(a):3]
(len(df.columns)-1)/4

#%%
from preprocessing.datasets import get_clean_dataset
df = get_clean_dataset(nb_min=30, dropna=True, from_disc=False)
df.shape

# %%
from utils.utils_graphic import plot_user_selection
plot_user_selection(2)
    