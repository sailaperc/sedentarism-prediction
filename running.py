# %%
import os
from experiments.experiment_running import run_all_experiments
import pickle as pkl



def fitness():
    pass


run_all_experiments(verbose=1)


# %%
# def f():
#     path_to_file = './pkl/experiments'
#     for fn in os.listdir(path_to_file):
#         nfn = fn[:-4] + '_per' + fn[-4:]
#         os.rename(f'{path_to_file}/{fn}', f'{path_to_file}/{nfn}')


# f()
# %%
