from sklearn.metrics import mean_squared_error, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from preprocessing.model_ready import split_x_y
from utils.utils import get_user_data

def per_user(df, model, model_type):
    assert (model_type == 'regression' or model_type == 'classification'), 'Not a valid model type.'
    #shorthand for ternary operator,
    scoring = ('mean_squared_error', 'f1_weighted')[model_type == 'classification']
    i = 0

    scores = []
    kfold = StratifiedKFold(n_splits=10)
    for userid in df.index.get_level_values(0).drop_duplicates():
        x, y = split_x_y(get_user_data(df, userid), model_type)
        x, y = x.values.astype('float32'), y.values.astype('float32')
        results = cross_val_score(model, x, y, cv=kfold, scoring=scoring)
        scores.append(results.mean())
        if i % 10 == 0:
            print('modelos sobre usuario ', i, ' finalizado.')
        i += 1
        del x
        del y

    return scores


def live_one_out(df, model, model_type):
    scoring_func = (mean_squared_error, f1_score)[model_type == 'classification']

    scores = []
    i = 0
    logo = LeaveOneGroupOut()
    groups = df.index.get_level_values(0)
    x, y = split_x_y(df, model_type)
    x, y = x.values.astype('float32'), y.values.astype('float32')
    for train_index, test_index in logo.split(x, y, groups):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        #f1.append(f1_score(y_test, y_pred, average='weighted'))
        scores.append(scoring_func(y_test, y_pred))
        if i % 10 == 0:
            print('modelos sobre usuario ', i, ' finalizado.')
        i += 1
        del x_train
        del x_test
        del y_train
        del y_test
    return scores
