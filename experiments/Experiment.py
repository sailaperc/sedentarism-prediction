from abc import ABC, abstractmethod
from preprocessing.datasets import get_lagged_dataset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, f1_score
from preprocessing.model_ready import split_x_y

class Experiment(ABC):
    def __init__(self, model, model_type, included_data, user, nb_lags, period, gran, need_3d_input):
        self.lags = nb_lags
        self.model_type = model_type
        self.included_data = included_data
        self.user = user
        self.nb_lags = nb_lags
        self.period = period
        self.gran = gran
        self.model = model 
        self.need_3d_input = need_3d_input
        self.dataset = get_lagged_dataset(model_type='regression', included_data='ws', user=-1, nb_lags=1, period=1, gran='1h')
        if need_3d_input:
            #x_test = x_test.reshape(x_test.shape[0], time_lags, number_of_features)

            

    @abstractmethod
    def run(self):
        pass

class PersonalExperiment(Experiment):
    def run(self):
        assert (self.model_type == 'regression' or self.model_type == 'classification'), 'Not a valid model type.'
        #shorthand for ternary operator,
        scoring = ('mean_squared_error', 'f1_weighted')[self.model_type == 'classification']
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

        return scores

class ImpersonalExperiment(Experiment):
    def run(self):
        scoring_func = (mean_squared_error, f1_score)[self.model_type == 'classification']
        scores = []
        i = 0
        logo = LeaveOneGroupOut()
        groups = df.index.get_level_values(0)
        x, y = split_x_y(df, model_type)
        x, y = x.values.astype('float32'), y.values.astype('float32')
        for train_index, test_index in logo.split(x, y, groups):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.model.fit(x_train, y_train)
            y_pred = self.model.predict(x_test)
            #f1.append(f1_score(y_test, y_pred, average='weighted'))
            scores.append(scoring_func(y_test, y_pred))
            if i % 10 == 0:
                print('modelos sobre usuario ', i, ' finalizado.')
            i += 1

            del x_train; del x_test; del y_train; del y_test; del y_pred
        del x; del y
        return scores

class HybridExperiment(Experiment):
    def run(self):
        pass