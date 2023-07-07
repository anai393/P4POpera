from skika.transfer import opera_wrapper
from matplotlib import pyplot as plt
import seaborn as sns
import itertools
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

num_phantom_branches= 50
obs_period = 500
min_obs_period = 1000
obs_window_size = 50

num_trees = 50
rf_lambda = 1
squashing_delta=7
perf_window_size=500
split_range=10
conv_delta=0.001
conv_threshold=0.15
#conv_threshold=0.05

rf_lambda = 1
squashing_delta=7
conv_delta=0.1
conv_threshold=0.15
perf_window_size=5000
split_range=10

def flatten(t):
    return [item for sublist in t for item in sublist]

class ClassifierMetrics:
    def __init__(self):
        self.correct = 0
        self.instance_idx = 0
        self.y_true = []
        self.y_pred= []

def log_metrics(count, sample_freq, metric, classifier, lst, y_p, y_t):
    if count % sample_freq == 0 and count != 0:
        accuracy = round(metric.correct / sample_freq, 2)

        # Phantom tree outputs
        f = int(classifier.get_full_region_complexity())
        e = int(classifier.get_error_region_complexity())
        c = int(classifier.get_correct_region_complexity())

        #print(f"{count},{accuracy},{f},{e},{c}")
        lst.append((count, accuracy))
        y_p.append(metric.y_pred)
        y_t.append(metric.y_true)
        metric.correct = 0
        metric.y_pred = []
        metric.y_true = []

    return lst, y_p, y_t

def get_cls_prc_rc(y_true, y_pred):
    pr, rec, f, sup = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1, 2,3,4,5], zero_division = 0)
    return pr, rec

def get_cls_acc(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    return acc

def get_cls_wise_acc(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    acc_cls = matrix.diagonal()/matrix.sum(axis=1)
    return list(acc_cls)

def print_fmt(avg, std):
    print_lst = []
    for i in range(len(avg)):
        cur_avg = "{:.2f}".format(avg[i])
        cur_std = "{:.2f}".format(std[i])
        cur_p = cur_avg + u"\u00B1" + cur_std
        print_lst.append(cur_p)
    return print_lst

def print_fmt_acc(avg, std):
    print_lst = "{:.2f}".format(avg)+ u"\u00B1" + "{:.2f}".format(std)
    return print_lst

def get_output(name, f, k, no_transfer, seed):
    if no_transfer == True:
        f = False
    data_file_path = name + " pre.arff;" + name + " post.arff";
    data_file_paths = data_file_path.split(";")
    result = []
    y_p = []
    y_t = []

    sample_freq = k
    random_state = seed
    force_disable_patching= no_transfer
    force_enable_patching = f
    grow_transfer_surrogate_during_obs=False

    classifier = opera_wrapper(
        len(data_file_paths),
        random_state,
        num_trees,
        rf_lambda,
        num_phantom_branches,
        squashing_delta,
        obs_period,
        conv_delta,
        conv_threshold,
        obs_window_size,
        perf_window_size,
        min_obs_period,
        split_range,
        grow_transfer_surrogate_during_obs,
        force_disable_patching,
        force_enable_patching)

    classifier_metrics_list = []
    for i in range(len(data_file_paths)):
        classifier.init_data_source(i, data_file_paths[i])
        classifier_metrics_list.append(ClassifierMetrics())

    classifier_idx = 0
    classifier.switch_classifier(classifier_idx)
    metric = classifier_metrics_list[classifier_idx]

    while True:
        if not classifier.get_next_instance():
            # Switch streams to simulate parallel streams

            classifier_idx += 1
            if classifier_idx >= len(data_file_paths):
                break

            classifier.switch_classifier(classifier_idx)
            metric = classifier_metrics_list[classifier_idx]

            print()
            print(f"switching to classifier_idx {classifier_idx}")
            continue

        classifier_metrics_list[classifier_idx].instance_idx += 1

        # test
        prediction = classifier.predict()
        actual_label = classifier.get_cur_instance_label()
        metric.y_pred.append(prediction)
        metric.y_true.append(actual_label)
        if prediction == actual_label:
            metric.correct += 1

        # train
        classifier.train()

        result, y_p, y_t = log_metrics(
            classifier_metrics_list[classifier_idx].instance_idx,
            sample_freq,
            metric,
            classifier,
            result, y_p, y_t)

    y_p = flatten(y_p)
    y_t = flatten(y_t)
    return result, y_p, y_t

def get_output_dif(name1, name2, f, k, transfer, seed):
    print("get_output_dif", name1, name2)
    if transfer == True:
        f = False
    data_file_path = name1 + ".arff;" + name2 + ".arff";
    data_file_paths = data_file_path.split(";")
    result = []
    y_p = []
    y_t = []

    sample_freq = k
    random_state = seed
    force_disable_patching= transfer
    force_enable_patching = f
    grow_transfer_surrogate_during_obs=False

    classifier = opera_wrapper(
        len(data_file_paths),
        random_state,
        num_trees,
        rf_lambda,
        num_phantom_branches,
        squashing_delta,
        obs_period,
        conv_delta,
        conv_threshold,
        obs_window_size,
        perf_window_size,
        min_obs_period,
        split_range,
        grow_transfer_surrogate_during_obs,
        force_disable_patching,
        force_enable_patching)

    classifier_metrics_list = []
    for i in range(len(data_file_paths)):
        classifier.init_data_source(i, data_file_paths[i])
        classifier_metrics_list.append(ClassifierMetrics())

    classifier_idx = 0
    classifier.switch_classifier(classifier_idx)
    metric = classifier_metrics_list[classifier_idx]

    while True:
        if not classifier.get_next_instance():
            # Switch streams to simulate parallel streams

            classifier_idx += 1
            if classifier_idx >= len(data_file_paths):
                break

            classifier.switch_classifier(classifier_idx)
            metric = classifier_metrics_list[classifier_idx]

            #print()
            #print(f"switching to classifier_idx {classifier_idx}")
            continue

        classifier_metrics_list[classifier_idx].instance_idx += 1

        # test
        prediction = classifier.predict()
        actual_label = classifier.get_cur_instance_label()
        metric.y_pred.append(prediction)
        metric.y_true.append(actual_label)
        if prediction == actual_label:
            metric.correct += 1

        # train
        classifier.train()

        result, y_p, y_t = log_metrics(
            classifier_metrics_list[classifier_idx].instance_idx,
            sample_freq,
            metric,
            classifier,
            result, y_p, y_t)

    y_p = flatten(y_p)
    y_t = flatten(y_t)
    return result, y_p, y_t

def get_cls_prc(name, k, f, dist):
    seed_lst = [0,1,2,3,4]
    no_avg_lst_rc = []
    no_avg_lst_prc = []
    yes_avg_lst_rc = []
    yes_avg_lst_prc = []
    for seed in seed_lst:
        acc_no_patch, y_p_no, y_t_no = get_output(name, f, k, True, seed)
        acc_yes_patch, y_p_yes, y_t_yes = get_output(name, f, k, False, seed)

        #print(acc_no_patch, len(y_p_no), len(y_t_no))

        no_prc, no_rc = get_cls_prc_rc(y_t_no, y_p_no)
        yes_prc, yes_rc = get_cls_prc_rc(y_t_yes, y_p_yes)

        no_avg_lst_prc.append(no_prc)
        no_avg_lst_rc.append(no_rc)

        yes_avg_lst_rc.append(yes_rc)
        yes_avg_lst_prc.append(yes_prc)

    print(name, k, f, dist)
    no_prc = np.mean(no_avg_lst_prc, axis=0)
    no_prc_std = np.std(no_avg_lst_prc, axis=0)
    no_print_prc = print_fmt(no_prc, no_prc_std)
    print("No transfer Precision:", no_print_prc)

    no_rc = np.mean(no_avg_lst_rc, axis=0)
    no_rc_std = np.std(no_avg_lst_rc, axis=0)
    no_print_rc = print_fmt(no_rc, no_rc_std)
    print("No transfer Recall:", no_print_rc)

    yes_prc = np.mean(yes_avg_lst_prc, axis=0)
    yes_prc_std = np.std(yes_avg_lst_prc, axis=0)
    yes_print_prc = print_fmt(yes_prc, yes_prc_std)
    print("Yes transfer Precision:", yes_print_prc)

    yes_rc = np.mean(yes_avg_lst_rc, axis=0)
    yes_rc_std = np.std(yes_avg_lst_rc, axis=0)
    yes_print_rc = print_fmt(yes_rc, yes_rc_std)
    print("Yes transfer Recall:", yes_print_rc)


    row_prc = []
    row_rc = []
    for i in range(len(no_print_prc)):
        row_prc.append(dist[i])
        row_prc.append(no_print_prc[i])
        row_prc.append(yes_print_prc[i])
    for i in range(len(no_print_rc)):
        row_rc.append(dist[i])
        row_rc.append(no_print_rc[i])
        row_rc.append(yes_print_rc[i])
    
    print("PRC row", row_prc)
    print("RC row", row_rc)

def get_cls_prc_dif(name1, name2, k, f, dist):
    print("get_cls_prc_dif", name1, name2)
    seed_lst = [0,1,2,3,4]
    no_avg_lst_rc = []
    no_avg_lst_prc = []
    yes_avg_lst_rc = []
    yes_avg_lst_prc = []
    for seed in seed_lst:
        acc_no_patch, y_p_no, y_t_no = get_output_dif(name1, name2, f, k, True, seed)
        acc_yes_patch, y_p_yes, y_t_yes = get_output_dif(name1, name2, f, k, False, seed)

        #print(acc_no_patch, len(y_p_no), len(y_t_no))

        no_prc, no_rc = get_cls_prc_rc(y_t_no, y_p_no)
        yes_prc, yes_rc = get_cls_prc_rc(y_t_yes, y_p_yes)

        no_avg_lst_prc.append(no_prc)
        no_avg_lst_rc.append(no_rc)

        yes_avg_lst_rc.append(yes_rc)
        yes_avg_lst_prc.append(yes_prc)


    no_prc = np.mean(no_avg_lst_prc, axis=0)
    no_prc_std = np.std(no_avg_lst_prc, axis=0)
    no_print_prc = print_fmt(no_prc, no_prc_std)
    print("No transfer Precision:", no_print_prc)

    no_rc = np.mean(no_avg_lst_rc, axis=0)
    no_rc_std = np.std(no_avg_lst_rc, axis=0)
    no_print_rc = print_fmt(no_rc, no_rc_std)
    print("No transfer Recall:", no_print_rc)

    yes_prc = np.mean(yes_avg_lst_prc, axis=0)
    yes_prc_std = np.std(yes_avg_lst_prc, axis=0)
    yes_print_prc = print_fmt(yes_prc, yes_prc_std)
    print("Yes transfer Precision:", yes_print_prc)

    yes_rc = np.mean(yes_avg_lst_rc, axis=0)
    yes_rc_std = np.std(yes_avg_lst_rc, axis=0)
    yes_print_rc = print_fmt(yes_rc, yes_rc_std)
    print("Yes transfer Recall:", yes_print_rc)


    row_prc = []
    row_rc = []
    for i in range(len(no_print_prc)):
        row_prc.append(dist[i])
        row_prc.append(no_print_prc[i])
        row_prc.append(yes_print_prc[i])
    for i in range(len(no_print_rc)):
        row_rc.append(dist[i])
        row_rc.append(no_print_rc[i])
        row_rc.append(yes_print_rc[i])
    
    print("PRC row", row_prc)
    print("RC row", row_rc)

def get_acc(name, k, f):
    seed_lst = [0,1,2,3,4]
    no_avg_lst_acc = []
    yes_avg_lst_acc = []
    for seed in seed_lst:
        acc_no_patch, y_p_no, y_t_no = get_output(name, f, k, True, seed)
        acc_yes_patch, y_p_yes, y_t_yes = get_output(name, f, k, False, seed)

        #print(acc_no_patch, len(y_p_no), len(y_t_no))

        no_acc = get_cls_acc(y_t_no, y_p_no)
        yes_acc = get_cls_acc(y_t_yes, y_p_yes)

        no_avg_lst_acc.append(no_acc)

        yes_avg_lst_acc.append(yes_acc)

    no_acc = np.mean(no_avg_lst_acc)
    no_acc_std = np.std(no_avg_lst_acc)

    no_print_acc = print_fmt_acc(no_acc, no_acc_std)
    print("No transfer Accuracy:", no_print_acc)

    yes_acc= np.mean(yes_avg_lst_acc)
    yes_acc_std = np.std(yes_avg_lst_acc)
    print(no_acc_std)
    yes_print_acc = print_fmt_acc(yes_acc, yes_acc_std)
    print("Yes transfer Accuracy:", yes_print_acc)

    
    print("ACC row", no_print_acc, yes_print_acc)

def get_acc_dif(name1, name2, k, f):
    print(name1, name2,k)
    seed_lst = [0,1,2,3,4]
    no_avg_lst_acc = []
    yes_avg_lst_acc = []
    for seed in seed_lst:
        acc_no_patch, y_p_no, y_t_no = get_output_dif(name1, name2, f, k, True, seed)
        acc_yes_patch, y_p_yes, y_t_yes = get_output_dif(name1, name2, f, k, False, seed)

        #print(acc_no_patch, len(y_p_no), len(y_t_no))
        print(seed, name1, name2)
        no_acc = get_cls_acc(y_t_no, y_p_no)
        yes_acc = get_cls_acc(y_t_yes, y_p_yes)

        no_avg_lst_acc.append(no_acc)

        yes_avg_lst_acc.append(yes_acc)

    no_acc = np.mean(no_avg_lst_acc)
    no_acc_std = np.std(no_avg_lst_acc)

    no_print_acc = print_fmt_acc(no_acc, no_acc_std)
    print("No transfer Accuracy:", no_print_acc)

    yes_acc= np.mean(yes_avg_lst_acc)
    yes_acc_std = np.std(yes_avg_lst_acc)
    print(no_acc_std)
    yes_print_acc = print_fmt_acc(yes_acc, yes_acc_std)
    print("Yes transfer Accuracy:", yes_print_acc)

    print("ACC row", no_print_acc, yes_print_acc)

def get_cls_avg_acc(name, k, f):
    seed_lst = [0,1,2,3,4]
    no_avg_lst_rc = []
    no_avg_lst_prc = []
    yes_avg_lst_rc = []
    yes_avg_lst_prc = []
    for seed in seed_lst:
        acc_no_patch, y_p_no, y_t_no = get_output(name, f, k, True, seed)
        acc_yes_patch, y_p_yes, y_t_yes = get_output(name, f, k, False, seed)

        #print(acc_no_patch, len(y_p_no), len(y_t_no))

        no_prc = get_cls_wise_acc(y_t_no, y_p_no)
        yes_prc = get_cls_wise_acc(y_t_yes, y_p_yes)
        # print(seed, name1, name2)

        no_avg_lst_prc.append(no_prc)
        yes_avg_lst_prc.append(yes_prc)


    no_prc = np.mean(no_avg_lst_prc, axis=0)
    no_prc_std = np.std(no_avg_lst_prc, axis=0)
    no_print_prc = print_fmt(no_prc, no_prc_std)
    print("No transfer Precision:", no_print_prc)


    yes_prc = np.mean(yes_avg_lst_prc, axis=0)
    yes_prc_std = np.std(yes_avg_lst_prc, axis=0)
    yes_print_prc = print_fmt(yes_prc, yes_prc_std)
    print("Yes transfer Precision:", yes_print_prc)

    row_prc = []
    for i in range(len(no_print_prc)):
        row_prc.append(no_print_prc[i])
        row_prc.append(yes_print_prc[i])



def get_cls_avg_acc_def(name1, name2, k, f):
    print("get_cls_avg_acc_def", name1, name2, k, f)
    seed_lst = [0,1,2,3,4]
    no_avg_lst_prc = []
    yes_avg_lst_prc = []
    for seed in seed_lst:
        acc_no_patch, y_p_no, y_t_no = get_output_dif(name1, name2, f, k, True, seed)
        acc_yes_patch, y_p_yes, y_t_yes = get_output_dif(name1, name2, f, k, False, seed)

        #print(acc_no_patch, len(y_p_no), len(y_t_no))
        print(seed, name1, name2)

        no_prc = get_cls_wise_acc(y_t_no, y_p_no)
        yes_prc = get_cls_wise_acc(y_t_yes, y_p_yes)

        no_avg_lst_prc.append(no_prc)
        yes_avg_lst_prc.append(yes_prc)


    no_prc = np.mean(no_avg_lst_prc, axis=0)
    no_prc_std = np.std(no_avg_lst_prc, axis=0)
    no_print_prc = print_fmt(no_prc, no_prc_std)
    print("No transfer Precision:", no_print_prc)


    yes_prc = np.mean(yes_avg_lst_prc, axis=0)
    yes_prc_std = np.std(yes_avg_lst_prc, axis=0)
    yes_print_prc = print_fmt(yes_prc, yes_prc_std)
    print("Yes transfer Precision:", yes_print_prc)

    print("k", k, "f", f)

    row_prc = []
    for i in range(len(no_print_prc)):
        row_prc.append(no_print_prc[i])
        row_prc.append(yes_print_prc[i])

    print("ACC row", row_prc)


arrowtown = [14398, 6328, 3636, 5109, 310, 19]
reefton = [5122, 1377, 399, 381, 29, 1]
materton = [2153, 876, 318, 584, 103, 14]
cromwell = [3250, 1217, 702, 2035, 435, 75]
invercargill = [5398, 1670, 463, 576, 24, 4]
dist_lst = [arrowtown, reefton, materton, cromwell, invercargill]   

name_lst = ['Arrowtown', 'Reefton', 'Masterton', 'Cromwell', 'Invercargill']
suffix = [" new temp", " new temp RH data"]
temp_names = []
temp_rh_names = []
for name in name_lst:
    temp_names.append(name+ suffix[0])
    temp_rh_names.append(name+ suffix[1])

step_size_lst = [10,100,1000]
f = True

for k in step_size_lst:
    for i in range(len(name_lst)):
        name = temp_rh_names[i]
        dist = dist_lst[i]
        print(name, dist)
        get_cls_prc(name, k , f, dist)
        print(name, k)
        get_acc(name, k , f)
        get_cls_avg_acc(name, k, f)


get_acc_dif(temp_rh_names[-2], temp_rh_names[-1], k, f)
for k in step_size_lst:
    for comb in itertools.permutations(temp_rh_names, 2):
        get_acc_dif(comb[0], comb[1], k, f)
        print("")


# f = True
# k = 100
# #get_cls_avg_acc_def(temp_rh_names[1], temp_rh_names[4], k, f)
# #get_cls_prc_dif(temp_rh_names[1], temp_rh_names[2], k, f, arrowtown)
# for comb in itertools.permutations(temp_rh_names, 2):
#     a = comb[1].split()[0]
#     print("Ass",a)
#     if a == 'Arrowtown':
#         dist = arrowtown
#     elif a == 'Reefton':
#         dist = reefton
#     elif a == 'Masterton':
#         dist = materton
#     elif a == 'Cromwell':
#         dist = cromwell
#     elif a == 'Invercargill':
#         dist = invercargill

#     #print(comb)
#     get_cls_prc_dif(comb[0], comb[1], k, f, dist)
#     #get_cls_avg_acc_def(comb[0], comb[1], k, f)
#     print("")

