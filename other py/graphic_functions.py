import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from utilfunction import get_user_data
import pandas as pd

def show_user_activity(df, user, mindate, maxdate, title=''):
    data = get_user_data(df, user)
    data = data.loc[(data.index.get_level_values(1) >= mindate) &
                    (data.index.get_level_values(1) < maxdate)]
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
