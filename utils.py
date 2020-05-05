import numpy
import pandas as pd
from scipy.stats.stats import pearsonr
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
import os

pd.options.mode.chained_assignment = None

numpy.random.seed(7)


def get_user_data(data, userId):
    """
    Get data of a specific user

    """
    try:
        return data.loc[data.index.get_level_values(0) == userId].copy()
    except KeyError:
        print('El usuario ', userId, ' no existe.')


def get_not_user_data(data, userId):
    """

    :return: all the data except that of the user specidied

    """
    try:
        return data.loc[data.index.get_level_values(0) != userId].copy()
    except KeyError:
        print('El usuario ', userId, ' no existe.')


def create_classifier_model(clf):
    '''
    Makes a pipeline from the clf param and a MinMaxScaler

    '''
    numeric_cols = ['numberOfConversations', 'wifiChanges',
                    'silenceLevel', 'voiceLevel', 'noiseLevel',
                    'hourSine', 'hourCosine',
                    'remainingminutes', 'pastminutes',
                    'distanceTraveled', 'locationVariance']
    transformer = ColumnTransformer([('scale', MinMaxScaler(), numeric_cols)],
                                    remainder='passthrough')
    return make_pipeline(transformer, clf)

def file_exists(file):
    return os.path.exists(file)









