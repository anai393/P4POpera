import numpy as np
import pandas as pd
import matplotlib
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import arff
from pandas2arff import pandas2arff

pd.options.mode.chained_assignment = None

def make_arff(name, col):
    new_lines = []
    with open(name+ ".arff") as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
        for l in lines:
            if len(l) != 0 :
                if l[0] != "@":
                    new_l = l.replace('"', '')
                    new_lines.append(new_l)
                else:
                    new_l = l.replace('real', 'numeric')
                    new_lines.append(new_l)

    b = ""
    if len(col) == 1:
        b = " new data"
    elif len(col) == 2:
        b = " new temp"
    else:
        b = " new temp RH data"

    with open(name + b + '.arff', 'w') as f:
        for item in new_lines:
            f.write("%s\n" % item)

def make_label(lst):
    label = [0]
    for i in range(1, len(lst)):
        pm = lst[i]
        if pm <= 12:
            label.append(0)
        elif pm > 12 and pm <= 35.4:
            label.append(1)
        elif pm > 35.4 and pm <= 55.4:
            label.append(2)
        elif pm > 55.4 and pm <= 150.4:
            label.append(3)
        elif pm > 150.4 and pm <= 250.4:
            label.append(4)
        else:
            label.append(5)
    label = [int(i) for i in label]
    return label

def make_dataset_old(name, chosen_sensors, chosen_times, col, center):
    start = chosen_times[0]
    end = chosen_times[1]
    time_duration = end - start
    df = pd.read_csv(name + ' filtered.csv')
    sensors = list(df["serialn"].unique())
    selected_sensors = []
    for s in sensors:
        for i in chosen_sensors:
            if i in s:
                selected_sensors.append(s)
    
    print(selected_sensors)
    dataset = []
    label = []
    for s in selected_sensors:
        new_pm = np.full(time_duration+1, np.nan)
        sensor_time = df.loc[ (df['serialn'] == s) & (df['Timestamp'] >= start) & (df['Timestamp'] <= end)]
        times = list(sensor_time.Timestamp)
        pm_2 = list(sensor_time[col])
        if (len(times)) == 0:
            a = df.loc[(df['serialn'] == s)]
            print(s)

        for i in range(len(times)):
            idx = int(times[i]) - start
            value = pm_2[i]
            #print(idx,value)
            new_pm[idx] = value
        dataset.append(new_pm)
        if s == center:
            label = new_pm
    label = make_label(label)

    dataset = np.transpose(dataset)
    new_df = pd.DataFrame(dataset, columns = selected_sensors)
    new_df["class"] = label
    new_df.dropna(inplace = True)
    new_df = new_df.tail(new_df.shape[0] -1)
    print(new_df.shape)
    #pandas2arff(new_df, name + '.arff')
    #make_arff(name)
    # arff.dump(name + '.arff'
    #   , new_df.values
    #   , relation= col
    #   , names=new_df.columns)
    new_df.to_csv(name + " new data.csv", index=True)
    

def make_dataset(name, chosen_sensors, chosen_times, col, center):
    feat_cnt = len(col)
    start = chosen_times[0]
    end = chosen_times[1]
    time_duration = end - start
    df = pd.read_csv(name + ' filtered.csv')
    sensors = list(df["serialn"].unique())
    selected_sensors = []
    col_names = []
    for s in sensors:
        for i in chosen_sensors:
            if i in s:
                selected_sensors.append(s)
                if len(col) > 1:
                    for c in col:
                        s = s.split()[0]
                        col_names.append(s+c)
                else:
                    s = s.split()[0]
                    col_names.append(s)
    
    print(selected_sensors)
    dataset = []
    label = []
    for s in selected_sensors:
        new_pm = np.full((time_duration+1,feat_cnt), np.nan)
        sensor_time = df.loc[ (df['serialn'] == s) & (df['Timestamp'] >= start) & (df['Timestamp'] <= end)]
        times = list(sensor_time.Timestamp)
        pm_2 = sensor_time[col]
        if (len(times)) == 0:
            a = df.loc[(df['serialn'] == s)]
            print(s)

        for i in range(len(times)):
            idx = int(times[i]) - start
            value = pm_2.iloc[[i]]
            #print(idx,value)
            new_pm[idx] = value
        
        for j in np.transpose(new_pm):
            dataset.append(j)
        if s == center:
            label = np.transpose(new_pm)[0]
    label = make_label(label)

    dataset = np.transpose(dataset)
    new_df = pd.DataFrame(dataset, columns = col_names)
    new_df["class"] = label
    new_df.dropna(inplace = True)
    new_df = new_df.tail(new_df.shape[0] -1)
    print(new_df.shape)
    pandas2arff(new_df, name + '.arff')
    make_arff(name,col)
    # arff.dump(name + '.arff'
    #   , new_df.values
    #   , relation= col
    #   , names=new_df.columns)
    if len(col) == 1:
        new_df.to_csv(name + " new data.csv", index=True)
    elif len(col) == 2:
        new_df.to_csv(name + " new temp data.csv", index=True)
    else:
        new_df.to_csv(name + " new temp RH data.csv", index=True)

def arrow_town(): #1min intervals
    name = 'Arrowtown'
    sensors = ["89996", "81166", "81901", "89558", "90143", "81935", "89566", "81802"]
    times = (75000, 132000)
    center = "ODIN-0207 (81166)"
    #make_dataset(name, sensors, times, ["PM2.5"], center)
    make_dataset(name, sensors, times, ["PM2.5", "Temperature"], center)
    make_dataset(name, sensors, times, ["PM2.5", "Temperature", "RH"], center)

#arrow_town()


def reefton(): #1min intervals
    name = 'Reefton'
    sensors = ["81083", "81224", "90432", "81190", "81810"]
    times = (0, 18000)
    center = "ODIN-0211 (81810)"
    #make_dataset(name, sensors, times, ["PM2.5"], center)
    make_dataset(name, sensors, times, ["PM2.5", "Temperature"], center)
    make_dataset(name, sensors, times, ["PM2.5", "Temperature", "RH"], center)

#reefton()

def masterton(): #5min intervals
    name = 'Masterton'
    #sensors = ['(58304)', '(58288)', '(58577)', '(58270)', '(58148)', '(58544)', '(58601)', '(58155)', '(58569)', '(58213)', '(58551)', '(58320)']
    #sensors = ['(58270)', '(58551)', '(58569)', '(58288)', '(58437)']
    #sensors = ['(58270)', '(58569)', '(58288)', '(58551)']
    sensors = ['(58270)', '(58569)', '(58288)']
    times = (0, 175000)
    center = "ODIN-SD-0315 (58270)"
    #make_dataset(name, sensors, times, ["PM2.5"], center)
    make_dataset(name, sensors, times, ["PM2.5", "Temperature"], center)
    make_dataset(name, sensors, times, ["PM2.5", "Temperature", "RH"], center)

#masterton()


def cromwell(): #1min intervals
    name = 'Cromwell'
    sensors = ['(81968)', '(90077)', '(89558)', '81869']
    times = (0, 50000)
    center = "ODIN-0157 (89558)"
    #make_dataset(name, sensors, times, ["PM2.5"], center)
    make_dataset(name, sensors, times, ["PM2.5", "Temperature"], center)
    make_dataset(name, sensors, times, ["PM2.5", "Temperature", "RH"], center)

#cromwell()


def invercargill(): #1min intervals
    name = 'Invercargill'
    sensors = ['ODIN-SD-0304 (58395)','ODIN-SD-0285 (58585)','ODIN-SD-0308 (58353)', 'ODIN-SD-0297 (58478)']
    times = (0, 20000)
    center = 'ODIN-SD-0308 (58353)'
    #make_dataset(name, sensors, times, ["PM2.5"], center)
    make_dataset(name, sensors, times, ["PM2.5", "Temperature"], center)
    make_dataset(name, sensors, times, ["PM2.5", "Temperature", "RH"], center)

arrow_town()
reefton()
masterton()
cromwell()
invercargill()


