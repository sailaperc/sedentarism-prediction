import matplotlib.pyplot as plt
from utils.utils import *
from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from numpy.random import seed
seed(7)

numeric_cols = ['cantConversation', 'wifiChanges',
                'stationaryCount', 'walkingCount', 'runningCount', 'silenceCount', 'voiceCount', 'noiseCount',
                'unknownAudioCount']
transformer = ColumnTransformer([('transformer', StandardScaler(),
                                  numeric_cols)],
                                remainder='passthrough')
model = make_pipeline(transformer, linear_model.LinearRegression())
df = pd.read_pickle('sedentarism.pkl')
mse1 = per_user_regression(df,model)
mse2 = live_one_out_regression(df,model)
plt.plot(mse1, label='per_user_regression')
plt.plot(mse2, label='live_one_out_regression')
plt.legend()
plt.show()
