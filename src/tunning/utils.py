import numpy as np
import pandas as pd

from skopt import load



def get_tunning_results(arch):
   
    cnn_hypers = ['Nr. filtros', 'Tam. kernel', 'Dropout (conv)', 'Nr. nodos densos', 'Dropout (denso)', 'Nr. épocas', 'Tam. bache', 'MSE']
    mlp_hypers = ['Nr. nodos', 'Nr. capas', 'Usar Batch Normalization', 'Dropout', 'Nr. épocas', 'Tam. bache', 'MSE']
    tcn_hypers = ['Nr. filtros', 'Tam. kernel', 'Dropout', 'Omitir conexiones', 'Usar Batch Normalization', 'Nr. épocas', 'Tam. bache', 'MSE']
    rnn_hypers = ['Nr. capas', 'Nr. unidades LSTM', 'Dropout (lstm)', 'Nr nodos densos', 'Dropout (denso)','Nr. épocas', 'Tam. bache', 'MSE']
    h = {
        'cnn' : cnn_hypers,
        'tcn': tcn_hypers,
        'mlp': mlp_hypers,
        'rnn': rnn_hypers
    }
   
    users = [34,32]
    pois = ['per', 'imp']

    results = []
    for poi in pois:
        for u in users:
            checkpoint_file = f'../../pkl/tunning/checkpoint_{arch}_{u}_{poi}.pkl'
            res = load(checkpoint_file)
            r = sorted(zip(res.func_vals, res.x_iters))[1]
            best = [arch, u, poi, *r[1], r[0]]
            results.append(best)

    df_results = pd.DataFrame(data=results)
    df_results.columns = ['arch', 'user', 'poi'] + h[arch]
    float_df = df_results.select_dtypes(include='float')
    float_cols = float_df.columns
    for col in h[arch]:
        if col in df_results.columns:
            df_results[col] = 2 ** df_results[col]
    df_results[float_cols] = np.round(float_df,3)
    df_results.style.hide_index()
    return  df_results