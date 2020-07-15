#%%
import numpy as np
import pandas as pd
from preprocessing.datasets import get_lagged_dataset, get_dataset
from utils.utils import get_user_data
from preprocessing.studentlife_raw import get_studentlife_dataset
#%%
df = get_lagged_dataset(nb_lags=4, period=1, model_type='classification')

# %%
a = get_user_data(df,9)

# %%
df