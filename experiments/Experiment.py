from abc import ABC, abstractmethod
from preprocessing.datasets import get_lagged_dataset

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

    def prepare_data():
        get_lagged_dataset(model_type='regression', included_data='ws', user=-1, nb_lags=1, period=1, gran='1h'):


    @abstractmethod
    def run(self):
        pass

class PersonalExperiment(Experiment):
