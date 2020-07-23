#%%
import numpy as np
import pandas as pd
from preprocessing.studentlife_raw import get_studentlife_dataset, get_sensor_data
from preprocessing.datasets import get_lagged_dataset


# %%
d =get_studentlife_dataset(60)


#%%
df = get_lagged_dataset()

#%%
df