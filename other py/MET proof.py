#chequear bien usuario 52
# probar sacando la cantidad de logs de cada actividad
from utilfunction import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.stats.stats import pearsonr
import math
#sns.set()
#sns.set_style("ticks")
#sns.despine()
df = pd.read_pickle('sedentarismdata.pkl')
df = delete_user(df,52)
df = METcalculation(df)
#df = delete_sleep_hours(df)

df['slevel'] = df['slevel'].astype('float')

def get_hour_labels():
    hours = []
    for h in range(0,24):
        if h<10:
            str = '0{0}:00'.format(h)
        else:
            str = '{0}:00'.format(h)
        hours.append(str)
    return hours


def show_graph(data, metric, user=-1):
    plt.close()
    if user>=0:
        dfuser = get_user_data(data, user)
    else: dfuser=data
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    userdata = dfuser.groupby(['dayofweek', 'hourofday'])['slevel']
    if metric=='Mean':
        userdata = userdata.mean()
        userdata = userdata.reset_index()
        userdata = userdata.pivot(index='dayofweek', values='slevel', columns='hourofday')
        sns.heatmap(userdata, vmin=1.3, cmap='RdBu_r')
    elif metric=='Standard Deviation':
        userdata = userdata.std()
        userdata = userdata.reset_index()
        userdata = userdata.pivot(index='dayofweek', values='slevel', columns='hourofday')
        sns.heatmap(userdata, vmin=0, cmap='autumn_r')
    plt.title('{0} activity of user {1}'.format(metric, user))
    plt.ylabel('Day of week')
    plt.xlabel('Hour of day')
    plt.yticks(np.arange(0.5,7.5), days, rotation='horizontal')
    plt.xticks(np.arange(0.5,24.5),get_hour_labels(), rotation='vertical')

    plt.show()


show_graph(df,'Mean')
show_graph(df, 'Standard Deviation')

show_graph(df,'Mean', 45)
show_graph(df,  'Standard Deviation', 45)

show_graph(df,'Mean', 25)
show_graph(df,  'Standard Deviation', 25)



for u in df.index.get_level_values(0).drop_duplicates():
    show_graph(df,'Mean', u)
    show_graph(df, 'Standard Deviation', u)

corrs = []
for u in df.index.get_level_values(0).drop_duplicates():
    fuser = get_user_data(df, u)
    fuser = fuser.groupby(['dayofweek', 'hourofday'])['slevel'].agg(['mean','std']).dropna()
    corr = pearsonr(fuser['mean'],fuser['std'])[0]
    if not math.isnan(corr):
        corrs.append(corr)

corrs = []
for u in df.index.get_level_values(0).drop_duplicates():
    fuser = get_user_data(df, u)
    met = float(fuser.groupby(['dayofweek', 'hourofday'])['slevel'].agg(['mean']).dropna().mean())
    corrs.append({'user': u,'metValue' : met})

ord = sorted(corrs, key=lambda k: k['metValue'])
or
#np.mean(corrs)
#0.8688055526065076
#np.std(corrs)
#0.052758920253513308

print('la correlacion de {0} es {1}'.format(44, corrs[44]))

"""
user = get_user_data(df, 41)
for m in np.arange(3,6):
    d = user.loc[user.index.get_level_values(1).month==m]
    print(d.shape)
    show_graph(d, 'Mean')
    show_graph(d, 'Standard Deviation')
"""