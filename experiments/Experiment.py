from abc import ABC, abstractmethod
from preprocessing.datasets import get_lagged_dataset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, f1_score
from preprocessing.model_ready import split_x_y
from utils.utils import get_user_data, get_not_user_data
from utils.utils import get_granularity_from_minutes
import pickle as pkl
from sklearn.preprocessing import StandardScaler
import time
from sklearn.metrics import mean_squared_error, auc
from utils.utils import file_exists


class Experiment(ABC):
    def __init__(self, model, model_type, included_data, user, nb_lags, period, nb_min, need_3d_input):
        self.lags = nb_lags
        self.task_type = model_type
        self.included_data = included_data
        self.user = user
        self.nb_lags = nb_lags
        self.period = period
        self.nb_min = nb_min
        self.model = model
        self.need_3d_input = need_3d_input
        self.validation_splits = 5
        self.experiment_data = {}
        self.train_data = None
        self.test_data = None
        self.gran = get_granularity_from_minutes(self.nb_min)

        if self.task_type == 'classification':
            self.scoring_func = auc
        else:
            self.scoring_func = mean_squared_error

    def time_series_split(self, train_data, test_data, n_splits):
        min_train = train_data.index.get_level_values(1).min()
        min_test = test_data.index.get_level_values(1).min()
        max_train = train_data.index.get_level_values(1).max()
        max_test = test_data.index.get_level_values(1).max()
        min_date = max([min_train, min_test])
        max_date = min([max_train, max_test])
        diff = max_date-min_date
        n_folds = n_splits + 1
        time_per_fold = diff / n_folds
        split_date = min_date
        for split_nb in range(n_splits):
            split_date = split_date + time_per_fold
            train_index = (train_data.index.get_level_values(1) <= split_date)
            train_data_split = train_data[train_index]
            if split_nb != n_splits-1:
                test_index_may = (
                    test_data.index.get_level_values(1) > split_date)
                test_index_inf = (test_data.index.get_level_values(
                    1) < (split_date+time_per_fold))
                test_index = test_index_may & test_index_inf
                test_data_split = test_data[test_index]
            else:
                test_index = (test_data.index.get_level_values(1) > split_date)
                test_data_split = test_data[test_index]
            X_train, y_train = split_x_y(train_data_split)
            X_test, y_test = split_x_y(test_data_split)
            yield X_train, y_train, X_test, y_test

    def prepare_data(self):
        self.dataset = get_lagged_dataset(model_type=self.task_type,
                                          included_data=self.included_data,
                                          user=self.user,
                                          nb_lags=self.nb_lags,
                                          period=self.period,
                                          nb_min=self.nb_min)

    def save(self):
        experiment_file = open(self.filename, 'wb')
        pkl.dump(self.experiment_data, experiment_file)
        experiment_file.close()
        print(f'Saved experiment in {self.filename}')

    def normalize(self, X_train, X_test):
        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_test = ss.transform(X_test)
        del ss
        return X_train, X_test

    def run(self):
        print('*** ' * 10)
        
        exp_name = f'{self.task_type}_{self.included_data}_gran{self.gran}_period{self.period}_lags{self.nb_lags}_model-{self.model.name}'

        self.filename = f'pkl/experiments/{exp_name}.pkl'

        if not file_exists(self.filename):
            print(f'Beginning experiment: ')
            print(exp_name)
            self.prepare_data()
            self.experiment_data['scores'] = []
            self.experiment_data['time_to_train'] = []
            self.experiment_data['nb_paramas'] = self.model.count_params()
            for split_data in self.time_series_split(self.train_data, self.test_data, self.validation_splits):
                
                X_train, y_train, X_test, y_test = split_data
                X_train, X_test = self.normalize(X_train, X_test) 

                if self.need_3d_input:
                    X_train = X_train.reshape(
                        X_train.shape[0], self.nb_lags, X_train.shape[0]/self.nb_lags)
                    X_test = X_test.reshape(
                        X_test.shape[0], self.nb_lags, X_test.shape[0]/self.nb_lags)

                start = time.time()
                self.model.fit(X_train, y_train)
                end = time.time()
                total = round((end - start) / 60, 3)
                self.experiment_data['time_to_train'].append(total)
                y_pred = self.model.predict(X_test)
                if self.task_type=='classification':
                    y_pred = (y_pred > 0.5) * 1.0
                
                print(y_pred.shape, y_test.shape)
                
                score = self.scoring_func(y_test, y_pred)
                self.experiment_data['scores'].append(score)
            print(f'Finishing experiment:  ')
            self.save()
        else:
            print('Experiment already done...passing')
            print(exp_name)
        
        print('*** ' * 10)


    def get_experiment_data(self):
        return self.experiment_data

    def get_results(self):
        return self.experiment_data['scores']

class PersonalExperiment(Experiment):

    def prepare_data(self):
        super().prepare_data()
        self.train_data = get_user_data(self.dataset, self.user)
        self.test_data = get_user_data(self.dataset, self.user)


class ImpersonalExperiment(Experiment):
    def prepare_data(self):
        super().prepare_data()
        self.train_data = get_not_user_data(self.dataset, self.user)
        self.test_data = get_user_data(self.dataset, self.user)


class HybridExperiment(Experiment):
    def prepare_data(self):
        super().prepare_data()
        self.train_data = self.dataset
        self.test_data = get_user_data(self.dataset, self.user)
