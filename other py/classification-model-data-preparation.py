import pandas as pd
import numpy as np

seed = 7
np.random.seed(seed)

s = pd.read_pickle('sedentarism.pkl')

s['slevel'] = ''
s.loc[s['isSedentary'] > 0.9999, 'slevel'] = 0 #'very sedentary'
s.loc[s['isSedentary'].between(0.9052, 0.9999), 'slevel'] = 1 #'sedentary'
s.loc[s['isSedentary'] < 0.9052, 'slevel'] = 2 #'less sedentary'

s = s.drop(columns=['audiomajor'])
#set type of numeric and categorical columns
numeric_cols = ['cantConversation', 'beforeNextDeadline', 'afterLastDeadline', 'hourofday', 'wifiChanges',
                'stationaryCount', 'walkingCount', 'runningCount', 'silenceCount', 'voiceCount', 'noiseCount',
                'unknownAudioCount', 'isSedentary', 'slevel']

for col in numeric_cols:
    s[col] = s[col].astype('float')

categorical_cols = ['partofday', 'dayofweek', 'activitymajor']
for col in categorical_cols:
    s[col] = s[col].astype('category')


swithdummies = pd.get_dummies(s.copy())
#se hace el shift para que el y de cada x corresponda al nivel de sedentarismo de una hora posterior
swithdummies = swithdummies.sort_index()
swithdummies['slevel'] = swithdummies['slevel'].shift(-1)

#se descartan las filas de x que correspondan a una hora sobre la que no haya informacion
# sobre el nivel de sedentarismo de la hora siguiente

for ind, row in swithdummies.iterrows():
    if not (ind[0], ind[1] + pd.DateOffset(hours=1)) in swithdummies.index:
        swithdummies.loc[(ind[0], ind[1])] = np.nan
swithdummies = swithdummies.dropna()


s = s.sort_values('isSedentary')
s = s.head(41397) #numero magico
s['slevel'] = pd.qcut(s['isSedentary'], 3, labels=['less sedentary', 'sedentary', 'very sedentary'])




features = [col for col in swithdummies.columns if 'slevel' != col]
X = swithdummies[features]
y = swithdummies['slevel']


X.to_pickle('classificationXsamples.pkl')
y.to_pickle('classificationysamples.pkl')