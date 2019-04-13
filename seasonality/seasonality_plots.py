from matplotlib import pyplot as plt
from matplotlib import colors
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import math

df = pd.read_pickle('./pkl/dataset.pkl')


def get_user_data(data, userId):
    try:
        return data.loc[data.index.get_level_values(0) == userId].copy()
    except KeyError:
        print('El usuario ', userId, ' no existe.')

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