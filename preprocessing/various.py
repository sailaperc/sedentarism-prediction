import pandas as pd
from utils import file_exists


def addSedentaryLevel(df, metValues=(1.3, 5, 8.3)):
    '''
    Calculates de metLevel feature from the metValues

    '''

    dfcopy = df.copy()
    metLevel = (dfcopy['stationaryLevel'] * metValues[0] +
                dfcopy['walkingLevel'] * metValues[1] +
                dfcopy['runningLevel'] * metValues[2])
    dfcopy['slevel'] = metLevel
    return dfcopy


def addSedentaryClasses(df, drop_slevel= True):
    """
    Generate an sclass column in the dataframe with true if sedentary and false if not sedentary

    """
    dfcopy = df.copy()
    dfcopy['sclass'] = ''
    dfcopy.loc[df['slevel'] >= 1.5, 'sclass'] = 0.0  # 'sedentary'
    dfcopy.loc[df['slevel'] < 1.5, 'sclass'] = 1.0  # 'not sedentary'
    # dfcopy['actualClass'] = dfcopy['sclass']
    if drop_slevel:
        dfcopy.drop(['slevel'], inplace=True, axis=1)
    return dfcopy


def makeDummies(df):
    '''
    Transforms categorical features into dummy features (one boolean feature for each categorical possible value)

    '''
    dfcopy = df.copy()
    categorical_cols = ['dayofweek', 'activitymajor']
    for col in categorical_cols:
        dfcopy[col] = dfcopy[col].astype('category')
    for col in set(df.columns) - set(categorical_cols):
        dfcopy[col] = dfcopy[col].astype('float')
    dummies = pd.get_dummies(dfcopy.select_dtypes(include='category'))
    dfcopy.drop(categorical_cols, inplace=True, axis=1)
    return pd.concat([dfcopy, dummies], axis=1, sort=False)


def delete_user(df, user):
    '''
    Deletes a specific user.

    '''
    return df.copy().loc[df.index.get_level_values(0) != user]


def delete_sleep_hours(df):
    return df.loc[(df['sclass'] >= 1.0) |
                  ((df.index.get_level_values(1).hour < 22) &
                   (df.index.get_level_values(1).hour > 5))]