from sklearn.model_selection import train_test_split
from utils.utils import get_user_data, get_not_user_data
from sklearn.preprocessing import StandardScaler
from preprocessing.datasets import get_lagged_dataset


def split_x_y(data, model_type):
    '''
    [for regression]
    Returns x and y for a lagged dataset, where 'y' is the column named 'slevel(t)', that is, the sedentary level of the present
    '''
    if model_type=='classification':
        target_feature = 'sclass'
    else:
        target_feature = 'slevel'

    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return x, y

def get_train_test_data(model_type,  nb_lags=1, period=1, gran='1h', user=-1, standarize=True):
    '''
    From a specific and already processed dataset, generated x_train, x_test, y_train, y_test.

    If standarize is true, X will be standarized
    '''
    assert (model_type == 'regression' or model_type == 'classification'), 'Not a valid model type.'
    data = get_lagged_dataset(model_type, nb_lags, period, gran, user)
    x, y = split_x_y(data, model_type)
    if user!=-1:
        x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, train_size=2 / 3)
    else:
        x_test = get_user_data(x, user)
        x_train = get_not_user_data(x, user)
        y_test = get_user_data(y, user)
        y_train = get_not_user_data(y, user)
    x_train, y_train = x_train.values.astype("float32"), y_train.values.astype("float32")
    x_test, y_test = x_test.values.astype("float32"), y_test.values.astype("float32")
    if standarize:
        numeric_cols = ['stationaryLevel', 'walkingLevel', 'runningLevel',
                        'numberOfConversations', 'wifiChanges',
                        'silenceLevel', 'voiceLevel', 'noiseLevel',
                        'hourSine', 'hourCosine',
                        'remainingminutes', 'pastminutes',
                        'locationVariance']
        to_standarize = [col + '(t-{0})'.format(lag) for lag in range(1, nb_lags + 1) for col in numeric_cols]
        # get_loc gets the number of a column based on its name
        to_standarize = [data.columns.get_loc(c) for c in to_standarize]
        ss = StandardScaler()
        x_train[:, to_standarize] = ss.fit_transform(x_train[:, to_standarize])
        x_test[:, to_standarize] = ss.transform(x_test[:, to_standarize])

    return x_train, y_train, x_test, y_test



