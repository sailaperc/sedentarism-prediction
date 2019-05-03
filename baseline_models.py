from os import listdir
from os.path import isfile, join
from sklearn.linear_model import LinearRegression
import pandas as pd
from utils import get_data, get_train_test_data
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

path = 'pkl/datasets'
files = listdir(path)

df = pd.DataFrame(columns=['gran','periods','lags','mae'])

for f in files:#files:
    print(f)
    i_lags = f.find('lags')
    i_period = f.find('period')
    i_gran = f.find('gran')
    i_punto = f.find('.')

    gran = f[i_gran + 4: i_period-1]
    period = int(f[i_period+6 : i_lags-1])
    lags = int(f[i_lags+4 : i_punto])

    user = 57

    print('lags :' + str(lags) + '. period: ' + str(period) + '. gran: ' + str(gran))

    data = get_data(False, lags, period, gran, user)
    x_train, y_train, x_test, y_test = get_train_test_data(user, True, lags, period, gran,True)
    lr = LinearRegression()
    lr.fit(x_train,y_train)
    s = 'lags :' + str(lags) + '. period: ' + str(period) +\
        '. gran: ' + str(gran) + '. score: ' + str(lr.score(x_test,y_test))
    new = pd.DataFrame(
        {'lags' : [lags],
         'period' : period,
         'gran' : gran,
         'mae' : float(round(mean_absolute_error(y_test, lr.predict(x_test)), 3))})
    df = df.append(new)



g = sns.FacetGrid(df, col='gran')
g.map(plt.scatter,'period','mae')
plt.show()

