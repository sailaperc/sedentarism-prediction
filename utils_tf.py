from sklearn.metrics import mean_squared_error, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_validate
from utils import get_X_y_regression,get_X_y_classification, get_user_data

def per_user_regression(df, model):
    print('per_user_regression')
    dfcopy = df.copy()
    mse = []
    for userid in df.index.get_level_values(0).drop_duplicates():
        X, y = get_X_y_regression(get_user_data(dfcopy, userid))
        kfold = StratifiedKFold(n_splits=10)
        results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
        mse.append(-results.mean())
        if userid % 10 == 0:
            print('modelos sobre usuario ', userid, ' finalizado.')
    return mse

def live_one_out_regression(df, model):
    print('live_one_out_regression')
    dfcopy = df.copy()
    mse = []
    i = 0
    logo = LeaveOneGroupOut()
    groups = df.index.get_level_values(0)
    X, y = get_X_y_regression(dfcopy)
    for train_index, test_index in logo.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse.append(mean_squared_error(y_test, y_pred))
        if i % 10 == 0:
            print('modelos sobre usuario ', i, ' finalizado.')
        i += 1
    return mse

def per_user_classification(df, model):
    print('per_user_classification')
    dfcopy = df.copy()
    scoring = ['f1_weighted']
    f1 = []
    kfold = StratifiedKFold(n_splits=10)
    for userid in df.index.get_level_values(0).drop_duplicates():
        X, y = get_X_y_classification(get_user_data(dfcopy, userid))
        results = cross_validate(model, X, y, cv=kfold, scoring=scoring)
        f1.append(results['test_f1_weighted'].mean())
        print('modelos sobre usuario ', userid, ' finalizado.')
    return f1

def live_one_out_classification(df, model):
    dfcopy = df.copy()
    print('live_one_out_classification')
    i = 0
    f1 = []
    logo = LeaveOneGroupOut()
    groups = df.index.get_level_values(0)
    X, y = get_X_y_classification(dfcopy)
    for train_index, test_index in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1.append(f1_score(y_test, y_pred, average='weighted'))
        if i % 10 == 0:
            print('modelos sobre usuario ', i, ' finalizado.')
        i += 1
    return f1