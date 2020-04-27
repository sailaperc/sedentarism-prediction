from utils import get_data
from sklearn.metrics import mean_absolute_error


def run_naive_approach(users=[50, 31, 4]):
    '''
    This function creates a naive model for the specified users.
    It predicts the sedentary level of an hour t as the sedentary level of the hour t-1

    :param users:
    :return:
    '''
    results = []
    for user in users:
        data = get_data(True, 1, 1, '1h', user)
        #last two columns are slevel(t-1) and slevel(t)
        arr = data.iloc[:, -2:].to_numpy()
        f = int(arr.shape[0] * 2 / 3)
        t = arr.shape[0] - 1
        arr = arr[f:t, :]

        y_pred = arr[:, 0]
        y_true = arr[:, 1]
        results.append({"user": user, "mae": mean_absolute_error(y_pred, y_true)})
    return results

print("Naive approach results for selected users:")
for r in run_naive_approach():
    print("user: {0} --> MAE: {1}".format(r['user'], r['mae']))
