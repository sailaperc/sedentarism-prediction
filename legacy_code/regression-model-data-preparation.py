import pandas as pd
import numpy as np

seed = 7
np.random.seed(seed)

s = pd.read_pickle('sedentarism.pkl')

s = s.drop(columns=['audiomajor'])
#set type of numeric and categorical columns
numeric_cols = ['cantConversation', 'beforeNextDeadline', 'afterLastDeadline', 'hourofday', 'wifiChanges',
                'stationaryCount', 'walkingCount', 'runningCount', 'silenceCount', 'voiceCount', 'noiseCount',
                'unknownAudioCount', 'slevel']

for col in numeric_cols:
    s[col] = s[col].astype('float')

categorical_cols = ['partofday', 'dayofweek', 'activitymajor']

for col in categorical_cols:
    s[col] = s[col].astype('category')

swithdummies = pd.get_dummies(s.copy())
features = [col for col in swithdummies.columns if 'slevel' != col]

swithdummies = swithdummies.sort_index()
swithdummies['slevel'] = swithdummies['slevel'].shift(-1)
for ind, row in swithdummies.iterrows():
    if not (ind[0], ind[1] + pd.DateOffset(hours=1)) in swithdummies.index:
        swithdummies.loc[(ind[0], ind[1])] = np.nan
swithdummies.dropna(inplace=True)


X = swithdummies[features]
y = swithdummies['slevel']

X.to_pickle('regressionXsamples.pkl')
y.to_pickle('regressionysamples.pkl')