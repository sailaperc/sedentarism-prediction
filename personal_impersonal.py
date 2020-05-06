from sklearn.metrics import mean_squared_error, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_validate
from preprocessing.model_ready import split_x_y
from utils import get_user_data
from sklearn.utils.multiclass import type_of_target

def per_user(df, model, model_type):
    assert (model_type == 'regression' or model_type == 'classification'), 'Not a valid model type.'
    print(f'per_user_{model_type}')

    #shorthand for ternary operator,
    scoring = ('mean_squared_error', 'f1_weighted')[model_type == 'classification']

    scores = []
    kfold = StratifiedKFold(n_splits=10)
    for userid in df.index.get_level_values(0).drop_duplicates():
        x, y = split_x_y(get_user_data(df, userid), model_type)
        results = cross_val_score(model, x.values.astype("float32"), y.values.astype("float32"), cv=kfold, scoring=scoring)
        scores.append(results.mean())
        if userid % 10 == 0:
            print('modelos sobre usuario ', userid, ' finalizado.')
    return scores


def live_one_out(df, model, model_type):
    scoring_func = (mean_squared_error, f1_score)[model_type == 'classification']

    print(f'live_one_out_{model_type}')
    mse = []
    i = 0
    logo = LeaveOneGroupOut()
    groups = df.index.get_level_values(0)
    x, y = split_x_y(df, model_type)
    for train_index, test_index in logo.split(x, y, groups):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        #f1.append(f1_score(y_test, y_pred, average='weighted'))
        mse.append(scoring_func(y_test, y_pred))
        if i % 10 == 0:
            print('modelos sobre usuario ', i, ' finalizado.')
        i += 1
    return mse
