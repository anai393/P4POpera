import numpy as np
import pandas as pd

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
        b = " new temp data"
    else:
        b = " new temp RH data"

    with open(name + b + '.arff', 'w') as f:
        for item in new_lines:
            f.write("%s\n" % item)

def split_arff(name, train):
    test = 1 - train
    with open(name+ ".arff") as file:
        lines = file.readlines()
        lines = [line for line in lines]
        data = []
        header = []
        d = False
        for l in lines:
            if len(l) != 0:
                if len(l) == 6:
                    header.append(l)
                    d = True
                elif d == True:
                    data.append(l)
                else:
                    header.append(l)
        train_cnt = int(train * len(data))
        train_data = data[:train_cnt]
        test_data = data[train_cnt:]
    
    with open(name + ' pre.arff', 'w') as f:
        for item in header:
            f.write(item)
        for item in train_data:
            f.write(item)
    with open(name + ' post.arff', 'w') as f:
        for item in header:
            f.write(item)
        for item in test_data:
            f.write(item)


train = 0.2
# name = "Arrowtown new temp RH data"
# split_arff(name, train)
name_lst = ['Arrowtown', 'Reefton', 'Masterton', 'Cromwell', 'Invercargill']
suffix = [" new temp", " new temp RH data", " new"]
for name in name_lst:
    split_arff((name+ suffix[0]), train)
    split_arff((name+ suffix[1]), train)