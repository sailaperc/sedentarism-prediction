from os import listdir
from sklearn.linear_model import LinearRegression
from preprocessing.model_ready import get_lagged_dataset, get_train_test_data_regression
from sklearn.metrics import mean_absolute_error
import warnings
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

path = 'pkl/datasets'
files = listdir(path)

df = pd.DataFrame(columns=['gran', 'period', 'lags', 'mae'])

tresd = False
for f in files:  # files:
    print(f)
    i_lags = f.find('lags')
    i_period = f.find('period')
    i_gran = f.find('gran')
    i_punto = f.find('.')

    gran = f[i_gran + 4: i_period - 1]
    period = int(f[i_period + 6: i_lags - 1])
    lags = int(f[i_lags + 4: i_punto])

    user = 57

    print('lags :' + str(lags) + '. period: ' + str(period) + '. gran: ' + str(gran))

    data = get_dataset(False, lags, period, gran, user)
    x_train, y_train, x_test, y_test = get_train_test_data_regression(user, True, lags,
                                                                      period, gran, True)
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    s = 'lags :' + str(lags) + '. period: ' + str(period) + \
        '. gran: ' + str(gran) + '. score: ' + str(lr.score(x_test, y_test))
    new = pd.DataFrame(
        {'lags': [lags],
         'period': period,
         'gran': gran,
         'mae': float(round(mean_absolute_error(y_test, lr.predict(x_test)), 3))})
    df = df.append(new)
