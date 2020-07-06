
#%%
import matplotlib.dates as mdates
from preprocessing.datasets import get_dataset, get_user_data
from utils.utils_graphic import plot_met_statistics,show_user_activity
from preprocessing.various import get_activity_levels, addSedentaryLevel
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_style("whitegrid")

#%%
df = get_dataset(delete_inconcitencies=False, from_disc=False)
