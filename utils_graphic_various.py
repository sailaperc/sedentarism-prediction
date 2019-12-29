import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from utils import get_user_data, get_dataset
import pandas as pd
import seaborn as sns
from matplotlib import colors

df = pd.read_pickle('./pkl/dataset_gran1h.pkl')

def show_user_activity(user, mindate='2013-03-27 04:00:00', maxdate='2013-06-01 3:00:00', title=''):
    data = get_user_data(df, user)
    data = data.loc[(data.index.get_level_values(1) >= mindate) &
                    (data.index.get_level_values(1) < maxdate)]
    print(data.shape)
    xlabels = mdates.date2num(data.index.get_level_values(1))
    diff = len(pd.date_range(mindate,maxdate,freq='h',closed='left'))- data.shape[0]
    if diff>0:
        print('Faltan {0} buckets!'.format(diff))
    r = data['walkingLevel'].values + data['runningLevel'].values
    w = data['walkingLevel'].values

    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.set_figheight(3)
    fig.set_figwidth(15)

    # ax.plot_date(xlabels, w, c='yellow', alpha=.4, fmt='-')
    # ax.plot_date(xlabels, r, c='red', alpha=.7,fmt='-')
    ax.fill_between(xlabels, w, 0, facecolor='yellow', alpha=1, label='Walking')
    ax.fill_between(xlabels, w, r, facecolor='red', alpha=1, label='Running')
    ax.fill_between(xlabels, r, 1, facecolor='black', alpha=.4, label='Stationary')

    # ax.format_xdata = mdates.AutoDateFormatter()
    ax.set_ylim(0, 1)
    ax.set_xlim(xlabels[0], xlabels[-1])
    ax.autoscale_view()
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%A'))
    ax.xaxis.set_minor_locator(mdates.HourLocator())

    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    fig.autofmt_xdate()
    ax.grid(True)
    ax.set_ylabel('Cumulative activity type (%)')
    ax.set_xlabel('Time')
    ax.set_title(title)
    ax.legend(loc='upper right')
    plt.show()
    # https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior

def get_hour_labels():
    hours = []
    for h in range(0,24):
        if h<10:
            str = '0{0}:00'.format(h)
        else:
            str = '{0}:00'.format(h)
        hours.append(str)
    return hours

def plot_heatmap(metric, user=-1):
    plt.close()
    if user>=0:
        dfuser = get_user_data(df, user)
    else: dfuser=df
    dfuser['hourofday'] = dfuser.index.get_level_values(1).hour
    dfuser['dayofweek'] = dfuser.index.get_level_values(1).dayofweek
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    userdata = dfuser.groupby(['dayofweek', 'hourofday'])['slevel']
    if metric=='mean':
        userdata = userdata.mean()
        userdata = userdata.reset_index()
        userdata = userdata.pivot(index='dayofweek', values='slevel', columns='hourofday')
        sns.heatmap(userdata, vmin=1.3, cmap='RdBu_r')
    elif metric=='std':
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

def plot_heatmaps_mean(users=[50,31,4]):
    df = get_dataset()
    metric = 'mean'

    df['hourofday'] = df.index.get_level_values(1).hour
    df['dayofweek'] = df.index.get_level_values(1).dayofweek
    #days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    days = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

    fig, axes = plt.subplots(
            nrows=1, ncols=4,
            figsize=(15,4.4),
            gridspec_kw={'width_ratios':[15,15,15,1]}
            )

    cbar_ax = axes[-1]

    for i in range(0,3):
        user = users[i]
        ax = axes[i]
        userdata = get_user_data(df,user)
        userdata = userdata.groupby(['dayofweek', 'hourofday'])['slevel']
        userdata = userdata.mean()
        userdata = userdata.reset_index()
        userdata = userdata.pivot(index='dayofweek', values='slevel', columns='hourofday')
        sns.heatmap(userdata,
                    ax=ax,
                    vmin=1.3, vmax=2,
                    cmap='RdBu_r',
                    cbar= True if i==2 else False,
                    linewidths=.05,
                    cbar_ax = cbar_ax if i==2 else None,
                    )
        ax.set_title('Usuario {0}'.format(user))
        ax.set_ylabel('')
        ax.set_xlabel('')
        plt.sca(ax)
        if i==0: plt.yticks(np.arange(0.5, 7.5), days, rotation='horizontal')
        else: ax.tick_params(labelleft = False, tick1On=False)
        plt.xticks(np.arange(0.5, 24.5),
                   get_hour_labels(),
                   rotation='vertical',
                   )
        ax.set_xlabel('Hora del día',fontsize   =10)
        ax.tick_params(axis='x', which='major', labelsize=8)


    #fig.text(0.5, 0, 'Hora del día', ha='center', fontsize=14)
    fig.text(0, .5, 'Día de la semana', va='center', rotation='vertical', fontsize=14)


    #plt.xlabel('Day of week', fontsize=18, labelpad=23)
    #plt.ylabel('Hour of day', fontsize=18, labelpad=45)
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(left=.5,bottom=.15,wspace=0,hspace=0)
    plt.show()

def plot_heatmaps_std(users=[50,31,4]):
    df = get_dataset()
    metric = 'mean'

    df['hourofday'] = df.index.get_level_values(1).hour
    df['dayofweek'] = df.index.get_level_values(1).dayofweek
    #days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    days = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

    fig, axes = plt.subplots(
            nrows=1, ncols=4,
            figsize=(15,4.4),
            gridspec_kw={'width_ratios':[15,15,15,1]}
            )



    cbar_ax = axes[-1]

    for i in range(0,3):
        user = users[i]
        ax = axes[i]
        userdata = get_user_data(df,user)
        userdata = userdata.groupby(['dayofweek', 'hourofday'])['slevel']

        userdata = userdata.std()
        userdata = userdata.reset_index()
        userdata = userdata.pivot(index='dayofweek', values='slevel', columns='hourofday')

        sns.heatmap(userdata,
                    ax=ax,
                    #vmin=1.3, vmax=2,
                    cmap='autumn_r',
                    cbar= True if i==2 else False,
                    linewidths=.05,
                    cbar_ax = cbar_ax if i==2 else None,
                    )
        ax.set_title('Usuario {0}'.format(user))
        ax.set_ylabel('')
        ax.set_xlabel('')
        plt.sca(ax)
        if i==0: plt.yticks(np.arange(0.5, 7.5), days, rotation='horizontal')
        else: ax.tick_params(labelleft = False, tick1On=False)
        plt.xticks(np.arange(0.5, 24.5),
                   get_hour_labels(),
                   rotation='vertical',
                   )
        ax.set_xlabel('Hora del día',fontsize   =10)
        ax.tick_params(axis='x', which='major', labelsize=8)


    #fig.text(0.5, 0, 'Hora del día', ha='center', fontsize=14)
    fig.text(0, .5, 'Día de la semana', va='center', rotation='vertical', fontsize=14)


    #plt.xlabel('Day of week', fontsize=18, labelpad=23)
    #plt.ylabel('Hour of day', fontsize=18, labelpad=45)
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(left=.5,bottom=.15,wspace=0,hspace=0)
    plt.show()

def plot_by_week(user):
    dfu = get_user_data(df, user).droplevel(0).loc[:, 'slevel']
    date = pd.date_range('2013-03-27 04:00:00', '2013-06-01 3:00:00', freq='h')
    d = pd.DataFrame(index=date)
    d['slevel'] = dfu
    d.isna().sum()
    # Prepare data
    d['month'] = [t.strftime('%b') for t in d.index]
    d['day'] = [t.strftime('%d') for t in d.index]
    d['dayofweek'] = [t.strftime('%w') for t in d.index]
    d['week'] = [t.strftime('%U') for t in d.index]

    d['date'] = pd.to_datetime(
        [t.strftime('2013-03-{0} %H:%M:%S.0000'.format(str(int(t.strftime('%w')) + 1))) for t in d.index])
    d['numdate'] = mdates.date2num(d.date)

    # d.ffill(inplace=True)
    week = d['week'].unique()
    # Prep Colors
    np.random.seed(20)
    mycolors = np.random.choice(list(colors.CSS4_COLORS.keys()), len(week), replace=False)
    # Draw Plot
    plt.close('all')
    fig = plt.figure(figsize=(16, 12), dpi=80)
    ax = fig.add_subplot(111)
    for i, m in enumerate(week):
        ax.plot('numdate', 'slevel', data=d.loc[d.week == m, :], color=mycolors[i], label=m)
    # Decoration
    ax.grid(True)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%A'))
    plt.ylim((1, 9))
    fig.autofmt_xdate()
    plt.gca().set(ylabel='$MET value$', xlabel='$Day$')
    plt.yticks(fontsize=12, alpha=.7)
    ax.legend(loc='upper right')
    plt.title("Student {0} energy expenditure along the season".format(user), fontsize=20)
    plt.show()

def plot_by_month(user):
    dfu = get_user_data(df, user).droplevel(0).loc[:, 'slevel']
    date = pd.date_range('2013-03-27 04:00:00', '2013-06-01 3:00:00', freq='h')
    d = pd.DataFrame(index=date)
    d['slevel'] = dfu
    d.isna().sum()
    d.ffill(inplace=True)

    # Prepare data
    d['month'] = [t.strftime('%b') for t in d.index]
    d['day'] = [t.strftime('%d') for t in d.index]
    d['date'] = pd.to_datetime([t.strftime('2013-03-%d %H:%M:%S.0000') for t in d.index])
    d['numdate'] = mdates.date2num(d.date)
    month = d['month'].unique()
    # Prep Colors
    # np.random.seed(20)
    # mycolors = np.random.choice(list(colors.CSS4_COLORS.keys()), len(month), replace=False)
    # Draw Plot
    mycolors = ['black', 'blue', 'green', 'red']
    plt.close('all')
    fig = plt.figure(figsize=(16, 12), dpi=80)
    ax = fig.add_subplot(111)
    for i, m in enumerate(month):
        ax.plot('numdate', 'slevel', data=d.loc[d.month == m, :], color=mycolors[i], label=m)
    # Decoration
    ax.grid(True)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%A'))
    fig.autofmt_xdate()
    plt.gca().set(ylabel='$MET value$', xlabel='$Day$')
    plt.yticks(fontsize=12, alpha=.7)
    ax.legend(loc='upper right')
    plt.title("Student {0} energy expenditure along the season".format(user), fontsize=20)
    plt.show()

def show_train_prediction(user, architecture):
    info = train_cache[user][architecture]
    model = models[user][architecture]['model']
    plt.close()
    plt.figure(figsize=(15, 4))
    plt.title('Datos de entrenamiento y predicciones para el usuario {0} con la arquitectura {1}'.format(user, architecture))
    y_pred = model.predict(info['x_train'])
    plt.plot(info['y_train'], label='Train')
    plt.plot(y_pred, label='Predicción')
    plt.axhline(y=1.5, color='r', linestyle=':', )
    plt.legend(loc='upper right')
    plt.savefig('Imagenes/{0}lags_user{1}_arch{2}__train.png'.format(time_lags,user, architecture))

def show_test_prediction(user, architecture):
    info = test_cache[user][architecture]
    model = models[user][architecture]['model']
    plt.close()
    plt.figure(figsize=(15, 4))
    #plt.title('Datos de testeo y predicciones para el usuario {0} con la arquitectura {1}'.format(user, architecture))
    y_pred = model.predict(info['x_test'])
    plt.plot(info['y_test'], label='Prueba')
    plt.plot(y_pred, label='Predicción')
    plt.ylabel('MET')
    plt.axhline(y=1.5, color='r', linestyle=':', )
    plt.legend(loc='upper right')
    plt.savefig('Imagenes/{0}lags_user{1}_arch{2}__test.png'.format(time_lags,user, architecture))

def show_history_loss(user, architecture):
    history = models[user][architecture]['history']
    plt.close()
    plt.title('Train loss vs. Test loss of user {0} with architecture {1}'.format(user, architecture))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig('Imagenes/{0}lags_user{1}_arch{2}_loss.png'.format(time_lags,user, architecture))

def generate_prediction_images():
    for user in users:
        for architecture in range(1, number_of_architectures + 1):
            print('*** Generando predicciones para el usuario {0} y la arquitectura {1}... ***'.format(user,
                                                                                                       architecture))
            show_train_prediction(user, architecture)
            show_test_prediction(user, architecture)
            show_history_loss(user, architecture)