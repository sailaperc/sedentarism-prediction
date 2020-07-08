#%%
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from utils.utils import get_user_data
from preprocessing.datasets import get_dataset
from preprocessing.various import get_activity_levels, addSedentaryLevel
import pandas as pd
import seaborn as sns
from matplotlib import colors
import seaborn as sns
sns.set_style("whitegrid")

def plot_user_activity(user, mindate='2013-03-27 04:00:00', maxdate='2013-06-01 3:00:00',  df=None):
    '''
    Plot the cumulative activity type of a specific user between mindate and maxdate
    The default mindate and mindate is too broad and does not work
    '''

    # to plot user 52's inconsistencies execute:
    # show_user_activity(52, '2013-05-22 00:00:00', '2013-05-28 23:59:59')

    title = 'Actividad por tipo a lo largo del tiempo'
    if df is None:
        df = get_dataset(delete_inconcitencies=False, from_disc=False)

    data = get_activity_levels(get_user_data(df, user))
    
    data = data.loc[(data.index.get_level_values(1) >= mindate) &
                    (data.index.get_level_values(1) < maxdate)]
    print(data.shape)
    true_date_range = data.index.get_level_values(1)
    date_range = pd.date_range(mindate, maxdate, freq='h', closed='left')
    none_dates = date_range.difference(true_date_range)
    print('Faltan {0} buckets!'.format(len(none_dates)))
    
    xlabels = mdates.date2num(true_date_range)


    r = data['walkingLevel'].values + data['runningLevel'].values
    w = data['walkingLevel'].values

    plt.close()
    fig = plt.figure(figsize=(15,3))
    ax = fig.add_subplot(111)

    for date in none_dates:
        plt.axvline(mdates.date2num(date))

    # ax.plot_date(xlabels, w, c='yellow', alpha=.4, fmt='-')
    # ax.plot_date(xlabels, r, c='red', alpha=.7,fmt='-')
    #ax.fill_between(xlabels, w, 0, facecolor='yellow', alpha=1, label='Walking')
    #ax.fill_between(xlabels, w, r, facecolor='red', alpha=1, label='Running')
    #ax.fill_between(xlabels, r, 1, facecolor='black', alpha=.4, label='Stationary')
    ax.fill_between(xlabels, w, 0, facecolor='yellow', alpha=1, label='Caminando')
    ax.fill_between(xlabels, w, r, facecolor='red', alpha=1, label='Corriendo')
    ax.fill_between(xlabels, r, 1, facecolor='black', alpha=.4, label='Estacionario')
    #ax.format_xdata = mdates.AutoDateFormatter()
    ax.set_ylim(0, 1)
    ax.set_xlim(xlabels[0], xlabels[-1])
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%A'))
    ax.xaxis.set_minor_locator(mdates.HourLocator())
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    
    ax.autoscale_view()
    fig.autofmt_xdate()
    ax.grid(True)
    ax.set_ylabel('Acumulación de actividad por tipo (%)')
    #ax.set_ylabel('Cumulative activity type (%)')
    ax.set_xlabel('Tiempo')
    #ax.set_xlabel('Time')
    ax.set_title(title)
    ax.legend(loc='upper right')
    plt.show()
    # https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior


def plot_user_activity_and_met(user, mindate='2013-03-27 04:00:00', maxdate='2013-06-01 3:00:00',  df=None):

    title = f'Actividad del usuario {user}'
    if df is None:
        df = get_dataset(delete_inconcitencies=False, from_disc=False)

    data = get_activity_levels(get_user_data(df, user))
    
    data = data.loc[(data.index.get_level_values(1) >= mindate) &
                    (data.index.get_level_values(1) < maxdate)]
    print(data.shape)
    true_date_range = data.index.get_level_values(1)
    date_range = pd.date_range(mindate, maxdate, freq='h', closed='left')
    none_dates = date_range.difference(true_date_range)
    print('Faltan {0} buckets!'.format(len(none_dates)))
    
    xlabels = mdates.date2num(true_date_range)


    r = data['walkingLevel'].values + data['runningLevel'].values
    w = data['walkingLevel'].values

    plt.close()
    fig, (ax2, ax) = plt.subplots(2, 1, figsize=(15,6), sharex='all')
    fig.suptitle(title, fontsize=16)

    for date in none_dates:
        ax.axvline(mdates.date2num(date))
        ax2.axvline(mdates.date2num(date))


  
    ax.fill_between(xlabels, w, 0, facecolor='yellow', alpha=1, label='Caminando')
    ax.fill_between(xlabels, w, r, facecolor='red', alpha=1, label='Corriendo')
    ax.fill_between(xlabels, r, 1, facecolor='black', alpha=.4, label='Estacionario')
    ax.set_ylim(0, 1)
    ax.set_xlim(xlabels[0], xlabels[-1])
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%A'))
    ax.xaxis.set_minor_locator(mdates.HourLocator())
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.autoscale_view()
    fig.autofmt_xdate()
    ax.grid(True)
    ax.set_ylabel('Acumulación de actividad por tipo (%)')
    ax.set_xlabel('Tiempo')
    ax.legend(loc='upper right')

    data = addSedentaryLevel(get_user_data(df, user)).slevel

    data = data.loc[(data.index.get_level_values(1) >= mindate) &
                    (data.index.get_level_values(1) < maxdate)]
    ax2.plot(xlabels, data.values)
    ax.set_ylabel('Acumulación de actividad por tipo (%)')
    ax.set_xlabel('Tiempo')
    fig
    plt.show()


def plot_met_statistics():
    
    df = get_dataset(delete_inconcitencies=False, from_disc=False)
    df = df.groupby(level=0)['slevel'].agg(['mean','std']).sort_values('mean').reset_index()

    plt.close()
    fig, (ax,ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex='all')
    sns.barplot(x="userId", y="mean", data=df, ax=ax, order=list(df.userId),
                palette=sns.color_palette("Paired", 10))
    plt.xticks(rotation='vertical')
    ax.set_xlabel("")
    ax.set_ylabel("Media")
    ax.set_title('Nivel del MET')

    sns.barplot(x="userId", y="std", data=df, ax=ax2, order=list(df.userId),
                palette=sns.color_palette("Paired", 10))
    ax2.set_ylabel("Desviación estándar")
    ax2.set_xlabel("User ID")
    plt.show()


def plot_buckets_per_user():
    df = get_dataset(delete_inconcitencies=False, from_disc=False)
    df_with_nulls = get_dataset(dropna=False, delete_inconcitencies=False, from_disc=False)
    plt.figure(figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.grid(axis='x')
    d = df.groupby(level=0).size().to_frame('disp')
    d2 = df_with_nulls.isnull().any(axis=1).groupby(level=0).sum().to_frame('drop')
    a = pd.concat([d,d2], axis=1).sort_values(by='disp')

    disp = a['disp'].values
    drop = a['drop'].values

    sticks = a.index.values 
    ind = np.arange(len(sticks))


    p1 = plt.bar(ind, disp,)
    p2 = plt.bar(ind, drop, bottom=disp)

    plt.title('Buckets disponibles por usuario')
    plt.xticks(ind, sticks, rotation='vertical')
    plt.xlabel('Id del usuario')
    plt.ylabel('Cantidad de buckets')
    plt.legend((p1[0], p2[0]), ('Disponibles', 'Descartados'))


def plot_met_distribution(df=None, user=-1, log_transform=False):
    if df is None:
        df = get_dataset()
    if user >= 0:
        df = get_user_data(df, user)
    data = df.slevel
    if log_transform:
        data = np.log1p(data)
    sns.distplot(data, hist=True, kde=False)
    title = 'Histograma sobre el nivel de MET'
    if user>0:
        title += f' del usuario {user}'
    plt.title(title)
    plt.xlabel('Nivel de MET')
    plt.ylabel('Cant. ocurrencias')


def get_hour_labels():
    hours = []
    for h in range(0, 24):
        if h < 10:
            str = '0{0}:00'.format(h)
        else:
            str = '{0}:00'.format(h)
        hours.append(str)
    return hours


def plot_heatmap(metric, user=-1):
    '''
    Plot a heatmap with entries for each day of week and hour of day combination.
    If user is not selected the whole dataset is used to make the dataset
    If a user is provided the heatmap is made based on the data of that particular user

    The metric can be 'mean' or 'std'

    '''
    df = get_lagged_dataset()
    plt.close()
    if user >= 0:
        dfuser = get_user_data(df, user)
    else:
        dfuser = df
    dfuser['hourofday'] = dfuser.index.get_level_values(1).hour
    dfuser['dayofweek'] = dfuser.index.get_level_values(1).dayofweek
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    userdata = dfuser.groupby(['dayofweek', 'hourofday'])['slevel']
    if metric == 'mean':
        userdata = userdata.mean()
        userdata = userdata.reset_index()
        userdata = userdata.pivot(index='dayofweek', values='slevel', columns='hourofday')
        sns.heatmap(userdata, vmin=1.3, cmap='RdBu_r')
    elif metric == 'std':
        userdata = userdata.std()
        userdata = userdata.reset_index()
        userdata = userdata.pivot(index='dayofweek', values='slevel', columns='hourofday')
        sns.heatmap(userdata, vmin=0, cmap='autumn_r')
    plt.title('{0} activity of user {1}'.format(metric, user))
    plt.ylabel('Day of week')
    plt.xlabel('Hour of day')
    plt.yticks(np.arange(0.5, 7.5), days, rotation='horizontal')
    plt.xticks(np.arange(0.5, 24.5), get_hour_labels(), rotation='vertical')

    plt.show()


def plot_heatmaps_mean(users=[50, 31, 4]):
    '''
    Plot heatmaps of 3 different user using the metric mean

    '''

    df = get_lagged_dataset()
    metric = 'mean'

    df['hourofday'] = df.index.get_level_values(1).hour
    df['dayofweek'] = df.index.get_level_values(1).dayofweek
    # days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    days = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

    fig, axes = plt.subplots(
        nrows=1, ncols=4,
        figsize=(15, 4.4),
        gridspec_kw={'width_ratios': [15, 15, 15, 1]}
    )

    cbar_ax = axes[-1]

    for i in range(0, 3):
        user = users[i]
        ax = axes[i]
        userdata = get_user_data(df, user)
        userdata = userdata.groupby(['dayofweek', 'hourofday'])['slevel']
        userdata = userdata.mean()
        userdata = userdata.reset_index()
        userdata = userdata.pivot(index='dayofweek', values='slevel', columns='hourofday')
        sns.heatmap(userdata,
                    ax=ax,
                    vmin=1.3, vmax=2,
                    cmap='RdBu_r',
                    cbar=True if i == 2 else False,
                    linewidths=.05,
                    cbar_ax=cbar_ax if i == 2 else None,
                    )
        ax.set_title('Usuario {0}'.format(user))
        ax.set_ylabel('')
        ax.set_xlabel('')
        plt.sca(ax)
        if i == 0:
            plt.yticks(np.arange(0.5, 7.5), days, rotation='horizontal')
        else:
            ax.tick_params(labelleft=False, tick1On=False)
        plt.xticks(np.arange(0.5, 24.5),
                   get_hour_labels(),
                   rotation='vertical',
                   )
        ax.set_xlabel('Hora del día', fontsize=10)
        ax.tick_params(axis='x', which='major', labelsize=8)

    # fig.text(0.5, 0, 'Hora del día', ha='center', fontsize=14)
    fig.text(0, .5, 'Día de la semana', va='center', rotation='vertical', fontsize=14)

    # plt.xlabel('Day of week', fontsize=18, labelpad=23)
    # plt.ylabel('Hour of day', fontsize=18, labelpad=45)
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(left=.5, bottom=.15, wspace=0, hspace=0)
    plt.show()


def plot_heatmaps_std(users=[50, 31, 4]):
    '''
    Plot incredibly beautiful heatmaps of 3 different user using the metric mean

    '''

    df = get_lagged_dataset()
    metric = 'mean'

    df['hourofday'] = df.index.get_level_values(1).hour
    df['dayofweek'] = df.index.get_level_values(1).dayofweek
    # days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    days = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

    fig, axes = plt.subplots(
        nrows=1, ncols=4,
        figsize=(15, 4.4),
        gridspec_kw={'width_ratios': [15, 15, 15, 1]}
    )

    cbar_ax = axes[-1]

    for i in range(0, 3):
        user = users[i]
        ax = axes[i]
        userdata = get_user_data(df, user)
        userdata = userdata.groupby(['dayofweek', 'hourofday'])['slevel']

        userdata = userdata.std()
        userdata = userdata.reset_index()
        userdata = userdata.pivot(index='dayofweek', values='slevel', columns='hourofday')

        sns.heatmap(userdata,
                    ax=ax,
                    # vmin=1.3, vmax=2,
                    cmap='autumn_r',
                    cbar=True if i == 2 else False,
                    linewidths=.05,
                    cbar_ax=cbar_ax if i == 2 else None,
                    )
        ax.set_title('Usuario {0}'.format(user))
        ax.set_ylabel('')
        ax.set_xlabel('')
        plt.sca(ax)
        if i == 0:
            plt.yticks(np.arange(0.5, 7.5), days, rotation='horizontal')
        else:
            ax.tick_params(labelleft=False, tick1On=False)
        plt.xticks(np.arange(0.5, 24.5),
                   get_hour_labels(),
                   rotation='vertical',
                   )
        ax.set_xlabel('Hora del día', fontsize=10)
        ax.tick_params(axis='x', which='major', labelsize=8)

    # fig.text(0.5, 0, 'Hora del día', ha='center', fontsize=14)
    fig.text(0, .5, 'Día de la semana', va='center', rotation='vertical', fontsize=14)

    # plt.xlabel('Day of week', fontsize=18, labelpad=23)
    # plt.ylabel('Hour of day', fontsize=18, labelpad=45)
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(left=.5, bottom=.15, wspace=0, hspace=0)
    plt.show()


def plot_by_week(user):
    '''
    Plot a users energy expenditure over the entire season for each week

    '''

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
    '''
    Plot a users energy expenditure over the entire season for each month

    '''
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

