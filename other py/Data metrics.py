from utilfunction import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_pickle('sedentarismdata.pkl')
df = delete_user(df,52)
df = METcalculation(df)
df = delete_sleep_hours(df)
df = makeSedentaryClasses(df)

plt.close()
a = df.groupby(df.index.get_level_values(0))['sclass'].apply(lambda x : np.sum(x==0)).values
b = df.groupby(df.index.get_level_values(0))['sclass'].apply(lambda x : np.sum(x==1)).values
c = df.groupby(df.index.get_level_values(0))['sclass'].count().values
a = a/c
b = b/c
users = np.arange(1,49)
plt.scatter(users, a)
plt.scatter(users, b)
plt.grid(True)
xlabels = df.index.get_level_values(0).drop_duplicates()
plt.xticks(users, xlabels, rotation='vertical')
plt.show()

