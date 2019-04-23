import matplotlib.pyplot as plt
from utils import *
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
import numpy
numpy.random.seed(7)

from sklearn.svm import SVC

def show_metric(title, ylabel, labels, data, markers=('s','^'), colors=('b','r')):
    users = np.arange(1, 49)
    userslabel = df.index.get_level_values(0).drop_duplicates()
    plt.close()
    for d,m,c in zip(data,markers,colors):
        plt.scatter(users, d, marker=m, c=c)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('User')
    plt.legend(labels,
               loc='best')
    plt.xticks(users, userslabel, rotation='vertical')
    plt.ylim(0.55,1)
    plt.grid(True)
    plt.show()

show_metric('',
            'F1-Score',
            ['Impersonal', 'Personal'],
            [pd.read_pickle('f1_imp_nn_withoutsleep'), pd.read_pickle('f1_p_nn_withoutsleep')])


'''
df = pd.read_pickle('sedentarismdata.pkl')
df = delete_user(df,52)
df = METcalculation(df)
df = delete_sleep_hours(df)
df = makeSedentaryClasses(df)
df = makeDummies(df)
df = shift_hours(df,1, 'classification')
df.drop(['hourofday'],
        axis=1, inplace=True)
'''

estimator = KerasClassifier(build_fn=baseline_model, epochs=30, batch_size=512, verbose=2)
modelnn = create_model(estimator)
estimator2 = KerasClassifier(build_fn=baseline_model, epochs=30, batch_size=32, verbose=2)
modelnn2 = create_model(estimator2)

clf = LogisticRegression(solver='liblinear', max_iter=400, class_weight='balanced')
model = create_model(clf)

'''
#f1_p_nn = per_user_classification(df, modelnn, True)
f1_p_logreg = per_user_classification(df, model, True)
#pd.to_pickle(f1_p_logreg,'f1_p_logreg_withsleep')

a = pd.read_pickle('f1_p_logreg_withsleep')
b = pd.read_pickle('f1_p_logreg_withoutsleep')


f1_imp_nn = live_one_out_classification(df, modelnn, True)
f1_imp_nn1 = pd.read_pickle('f1_imp_nn_withsleep')
f1_imp_nn2 = pd.read_pickle('f1_imp_nn_withoutsleep')

pd.to_pickle(f1_imp_nn1,'f1_imp_nn_withoutsleep')
pd.to_pickle(f1_imp_nn2,'f1_imp_nn_withsleep')

f1_imp_logreg = live_one_out_classification(df, model, True)

b = df.groupby(df.index.get_level_values(0))['sclass'].apply(lambda x : np.sum(x==1)).values
c = df.groupby(df.index.get_level_values(0))['sclass'].count().values
b = b/c

"""
show_metric('Model F1-score for impersonal models ',
            'F1-score',
            ['nn', 'logreg'],
            [f1_imp_nn, f1_imp_logreg])
"""

show_metric('',
            'F1-Score',
            ['Impersonal', 'Personal'],
            [pd.read_pickle('f1_imp_nn_withsleep'), pd.read_pickle('f1_p_logreg_withsleep')])

show_metric('',
            'F1-Score',
            ['Impersonal', 'Personal'],
            [pd.read_pickle('f1_imp_nn_withoutsleep'), pd.read_pickle('f1_p_logreg_withoutsleep')])
'''
print('f1_imp_nn_withsleep')
print(np.mean(pd.read_pickle('f1_imp_nn_withsleep')))
print(np.std(pd.read_pickle('f1_imp_nn_withsleep')))
print('f1_imp_nn_withoutsleep')
print(np.mean(pd.read_pickle('f1_imp_nn_withoutsleep')))
print(np.std(pd.read_pickle('f1_imp_nn_withoutsleep')))
print('f1_imp_logreg_withsleep')
print(np.mean(pd.read_pickle('f1_imp_logreg_withsleep')))
print(np.std(pd.read_pickle('f1_imp_logreg_withsleep')))
print('f1_imp_logreg_withoutsleep')
print(np.mean(pd.read_pickle('f1_imp_logreg_withoutsleep')))
print(np.std(pd.read_pickle('f1_imp_logreg_withoutsleep')))
print('f1_p_logreg_withsleep')
print(np.mean(pd.read_pickle('f1_p_logreg_withsleep')))
print(np.std(pd.read_pickle('f1_p_logreg_withsleep')))
print('f1_p_logreg_withoutsleep')
print(np.mean(pd.read_pickle('f1_p_logreg_withoutsleep')))
print(np.std(pd.read_pickle('f1_p_logreg_withoutsleep')))
print('f1_p_nn_withsleep')
print(np.mean(pd.read_pickle('f1_p_nn_withsleep')))
print(np.std(pd.read_pickle('f1_p_nn_withsleep')))
print('f1_p_nn_withoutsleep')
print(np.mean(pd.read_pickle('f1_p_nn_withoutsleep')))
print(np.std(pd.read_pickle('f1_p_nn_withoutsleep')))


df = pd.read_pickle('sedentarismdata.pkl')
df = delete_user(df,52)
df = METcalculation(df)
df = makeSedentaryClasses(df)
df = makeDummies(df)
df = shift_hours(df,1, 'classification')
df.drop(['hourofday'],
        axis=1, inplace=True)
#pd.to_pickle(live_one_out_classification(df, modelnn),'f1_imp_nn_withsleep')
pd.to_pickle(live_one_out_classification(df, model),'f1_imp_logreg_withsleep')
#pd.to_pickle(per_user_classification(df, model),'f1_p_logreg_withsleep')
#data = df.loc[ (df.index.get_level_values(0) >= 50) & (df.index.get_level_values(0)<60) ]
#pd.to_pickle(per_user_classification(data, modelnn2),'f1_p_nn_withsleep_6')

df = pd.read_pickle('sedentarismdata.pkl')
df = delete_user(df,52)
df = METcalculation(df)
df = delete_sleep_hours(df)
df = makeSedentaryClasses(df)
df = makeDummies(df)
df = shift_hours(df,1, 'classification')
df.drop(['hourofday'],
        axis=1, inplace=True)
#pd.to_pickle(live_one_out_classification(df, modelnn),'f1_imp_nn_withoutsleep')
#pd.to_pickle(live_one_out_classification(df, model),'f1_imp_logreg_withoutsleep')
#pd.to_pickle(per_user_classification(df, model),'f1_p_logreg_withoutsleep')
data = df.loc[ (df.index.get_level_values(0) >= 50) & (df.index.get_level_values(0)<60) ]
pd.to_pickle(per_user_classification(data, modelnn2),'f1_p_nn_withoutsleep_5')


def join_pkls(name):
    f1 = []
    for i in np.arange(1,6):
        f1.extend(pd.read_pickle(name + '_' + str(i)))
    pd.to_pickle(f1,name)

join_pkls('f1_p_nn_withsleep')
join_pkls('f1_p_nn_withoutsleep')

pkl = pd.read_pickle('f1_p_nn_withoutsleep')
np.mean(pkl)


a = pd.read_pickle('f1_imp_nn_withoutsleep')
b = pd.read_pickle('f1_p_nn_withoutsleep')

plt.boxplot(a)
plt.show()
c = df.groupby(df.index.get_level_values(0))['noiseLevel'].count().values

data = pd.DataFrame(data={'f1':a,
                          'count':c})

plt.close()
sns.lmplot(x='count',y='f1', data=data)
plt.show()