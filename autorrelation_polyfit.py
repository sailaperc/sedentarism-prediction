from pandas import Series
from matplotlib import pyplot as plt
from numpy import polyfit
import pandas as pd
import numpy as np
import seaborn as sea

def get_user_data(data, userId):
    try:
        return data.loc[data.index.get_level_values(0) == userId].copy()
    except KeyError:
        print('El usuario ', userId, ' no existe.')

df = pd.read_pickle('./pkl/dataset.pkl')

for i in df.index.get_level_values(0).drop_duplicates():
    userdata = get_user_data(df, i).iloc[0:500]
    print(i)
    series = userdata.slevel
    # fit polynomial: x^2*b1 + x*b2 + ... + bn
    X = userdata.index.get_level_values(1).hour
    y = series.values

    degree = 5
    coef = polyfit(X, y, degree)
    print('Coefficients: %s' % coef)
    # create curve
    curve = list()
    for i in range(len(X)):
        value = coef[-1]
        for d in range(degree):
            value += X[i]**(degree-d) * coef[d]
        curve.append(value)
    # plot curve over original data
    plt.close()
    plt.plot(series.values,color='blue',linewidth=1)
    plt.plot(curve, color='red', linewidth=1)
    plt.show()
