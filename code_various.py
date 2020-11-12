#%% 
from experiments.experiments_results import get_experiments_data, generate_df_from_experiments
import matplotlib.pyplot as plt 
import numpy as np
df = get_experiments_data(with_y_test_pred=True)

def get_test_predicted_arrays(exp_data):
    zipped = zip(*exp_data)
    l = list(zipped)
    l[1] = [np.squeeze(a) for a in l[1]]
    y_test = np.concatenate(l[0]) 
    y_pred = np.concatenate(l[1]) 
    return y_test, y_pred

def print_results(fromi=1, toi=5, archs=['rnn', 'tcn', 'cnn', 'mlp'], poi='per', user=32, lags=1, period=1, gran=60):
    exp = df.loc[((df.poi==poi) & (df.user==user) & (df.nb_lags==lags) & (df.period==period) & (df.gran==gran))]
    plt.close()
    width = 2 + 2*(toi-fromi+1)
    plt.figure(figsize=(width,4))
    print(width)
    first_pass = True
    for arch in archs:
        exp_arch = exp.loc[df.arch==arch,:]
        exp_data = exp_arch.y_test_pred.values[0][fromi-1:toi]
        y_test, y_pred = get_test_predicted_arrays(exp_data)
        lw = .6
        if first_pass:
            plt.plot(y_test, label='Test', lw=lw)
            first_pass = False
        plt.plot(y_pred, label=f'Predicho ({arch})', lw=lw)
    plt.legend()
    plt.show()

print_results(fromi=1, toi=5, loop_over={'period':[1,4],'archs':['tcn']}, poi='imp', lags=8, user=12)
