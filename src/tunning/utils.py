#%%
import sys
sys.path.insert(0, "c:\\Users\\marsa\\Documents\\projects\\tesis-project\\src")
print(sys.path[0])

#%%
import pandas as pd
from skopt import load

#%%
def fitness():
    pass
for poi in ['per','imp']:
    for u in [32,34]:
        for arch in ['cnn','tcn','rnn','mlp']:
            filename = f'../../pkl/tunning/checkpoint_{arch}_{u}_{poi}.pkl'
            res = load(filename)
            x0 = res.x_iters
            y0 = res.func_vals
            print(len(x0))
            sorted(zip(y0, x0))

#%%
res = load('../../pkl/tunning/checkpoint_cnn_32_imp.pkl')
#%%
x0 = res.x_iters
y0 = res.func_vals
print(len(x0))
sorted(zip(y0, x0))
