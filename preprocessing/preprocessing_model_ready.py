from sklearn.model_selection import train_test_split
from utils import get_user_data, get_not_user_data
import pandas as pd
from sklearn.preprocessing import StandardScaler

def get_lagged_dataset(personal, nb_lags, period, gran, user=-1):
    '''
    Get a specific and already generated dataset based on nb_lags, period, gran.
    If personal is true, only returns the specific users data

    '''
    data = pd.read_pickle('pkl/datasets/gran{0}_period{1}_lags{2}.pkl'.format(gran, period, nb_lags))
    if personal and user != -1:
        return get_user_data(data, user)
    else:
        return data

def get_X_y_regression(df):
    """
    Separe data between X (feature) and y (prediction_variable)

    """
    dfcopy = df.copy()
    features = [col for col in dfcopy.columns if 'slevel' != col]
    return dfcopy[features].reset_index(drop=True), dfcopy['slevel'].reset_index(drop=True)

def get_X_y_classification(df, withActualClass=True):
    '''

    :param withActualClass: If the actual class should be used as a feature. Default is true
    '''
    dfcopy = df.copy()
    # if not withActualClass:
    #    dfcopy.drop(['actualClass'], inplace=True, axis=1)
    features = [col for col in dfcopy.columns if 'sclass' != col]
    return dfcopy[features].reset_index(drop=True), dfcopy['sclass'].reset_index(drop=True)


def get_train_test_data_regression(user, standarize, lags, period, gran, personal):
    '''
    From a specific and already processed dataset, generated x_train, x_test, y_train, y_test.

    If standarize is true, X will be standarized
    '''
    data = get_lagged_dataset(personal, lags, period, gran, user)
    numeric_cols = ['stationaryLevel', 'walkingLevel', 'runningLevel',
                    'numberOfConversations', 'wifiChanges',
                    'silenceLevel', 'voiceLevel', 'noiseLevel',
                    'hourSine', 'hourCosine',
                    'remainingminutes', 'pastminutes',
                    'locationVariance']
    to_standarize = [col + '(t-{0})'.format(lag) for lag in range(1, lags + 1) for col in numeric_cols]
    # aux = to_standarize.copy()
    # aux.append('slevel(t)')
    # data = data.loc[:,aux]
    x, y = split_x_y_regression(data)
    if personal:
        x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, train_size=2 / 3)
    else:
        x_test = get_user_data(x, user)
        x_train = get_not_user_data(x, user)
        y_test = get_user_data(y, user)
        y_train = get_not_user_data(y, user)
    x_train, y_train = x_train.values.astype("float32"), y_train.values.astype("float32")
    x_test, y_test = x_test.values.astype("float32"), y_test.values.astype("float32")
    if standarize:
        # get_loc gets the number of a column based on its name
        to_standarize = [data.columns.get_loc(c) for c in to_standarize]
        ss = StandardScaler()
        x_train[:, to_standarize] = ss.fit_transform(x_train[:, to_standarize])
        x_test[:, to_standarize] = ss.transform(x_test[:, to_standarize])
    return x_train, y_train, x_test, y_test

def split_x_y_regression(data):
    '''
    [for regression]
    Returns x and y for a lagged dataset, where 'y' is the column named 'slevel(t)', that is, the sedentary level of the present

    '''
    x = data.iloc[:, [data.columns.get_loc(c) for _, c in enumerate(data.columns) if c != 'slevel(t)']]
    y = data.iloc[:, data.columns.get_loc('slevel(t)')]
    return x, y
