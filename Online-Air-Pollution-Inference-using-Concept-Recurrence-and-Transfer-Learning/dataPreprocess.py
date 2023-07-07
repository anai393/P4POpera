import numpy as np
import pandas as pd
import matplotlib
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

pd.options.mode.chained_assignment = None


def get_most_common(lst):
    difference = []
    for i in range(1,len(lst)):
        cur_dif = lst[i] - lst[i - 1]
        difference.append(cur_dif)
    common_3 = Counter(difference).most_common(3)
    return common_3

def filter(lst, diff_min, diff_max):
    diff_min = diff_min 
    diff_max = diff_max
    filtered = [[]]
    difference = []
    for i in range(len(lst)):
        if i == 0:
            filtered[-1].append(lst[0])
        else:
            cur_dif = lst[i] - lst[i - 1]
            difference.append(cur_dif)
            if cur_dif > diff_max or cur_dif < diff_min:
                new_group = [lst[i]]
                filtered.append(new_group)
            else:
                filtered[-1].append(lst[i])
    return filtered, difference

def preprocess(name):
    df = pd.read_csv(name + ' data.csv')
    sensors = list(df["serialn"].unique())
    # print(sensors)
    print("Number of sensors in " + name,len(sensors))
    sensor_time_lst = []
    z = 0
    for s in sensors:
        sensor_time = df.loc[df['serialn'] == s]
        boolean = not sensor_time["date"].is_unique
        #print("Has dups?", boolean, s, sensor_time.shape)
        if boolean:
            sensor_time.drop_duplicates(subset ="date",
                            keep = False, inplace = True)
            boolean = not sensor_time["date"].is_unique
            #print("Has dups?", boolean, sensor_time.shape)
        z += len(sensor_time)
        if len(sensor_time) > 5:
            sensor_time_lst.append(sensor_time)
    df = pd.concat(sensor_time_lst)

    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date',ascending=True, inplace=True)
    df.reset_index(inplace = True, drop = True)
    dates = list(df.date)
    start = dates[0]
    dates_norm = []
    for i in dates:
        totsec = (i-start).total_seconds()
        a = totsec/60
        dates_norm.append(a)
    df['Timestamp'] = dates_norm
    df.to_csv(name + " filtered.csv", index=False)

    plt_1 = plt.figure(figsize=(120, 20))
    sensors = list(df["serialn"].unique())
    sensor_sizes = []
    y_cnt = 1
    sensors_10 = []
    sensors_5 = []
    for s in sensors:
        sensor_df = df.loc[df['serialn'] == s]
        sensor_df.sort_values(by='Timestamp',ascending=True, inplace=True)
        times = list(sensor_df.Timestamp)
        most_common = get_most_common(times)
        sensor_start = times[0]
        sensor_end = times[-1]
        min_freq = most_common[0][0] - 1
        max_freq = most_common[0][0] + 1
        size = 0
        if most_common[0][0] == 10:
            sensors_10.append(s)
        elif most_common[0][0] == 5:
            sensors_5.append(s)
        elif most_common[0][0] == 1:
            filtered, difference = filter(times, min_freq,max_freq)
            for i in range(len(filtered)):
                x = filtered[i]
                size = size + len(x)
                #if len(x) > 100 and x[0]> 75000 and x[0] < 150000:
                if len(x) > 100:
                    y = [s]*len(x)
                    plt.plot(x,y, label= s)
        sensor_sizes.append((s, size))
        y_cnt += 1
    print(sensors_10)
    print(sensors_5)
    sensor_sizes.sort(key=lambda x: x[1])
    print(sensor_sizes)
    plt.savefig(name + 'all sensor plt.png')
    plt.show()

# name = 'Arrowtown'
# name ='Reefton'
# name = 'Masterton'
# name = 'Cromwell'
name = 'Invercargill'

preprocess(name)



