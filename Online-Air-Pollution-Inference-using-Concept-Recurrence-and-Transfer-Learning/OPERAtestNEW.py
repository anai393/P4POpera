from skika.transfer import opera_wrapper
from matplotlib import pyplot as plt
import seaborn as sns
import itertools


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



class ClassifierMetrics:
    def __init__(self):
        self.correct = 0
        self.instance_idx = 0

def log_metrics(count, sample_freq, metric, classifier, lst, d_size):
    if count % sample_freq == 0 and count != 0:
        accuracy = round(metric.correct / sample_freq, 2)

        # Phantom tree outputs
        f = int(classifier.get_full_region_complexity())
        e = int(classifier.get_error_region_complexity())
        c = int(classifier.get_correct_region_complexity())

        #print(f"{count},{accuracy},{f},{e},{c}")
        lst.append((count, accuracy))
        metric.correct = 0

    elif count == d_size:
        accuracy = round(metric.correct / (count%sample_freq), 2)

        # Phantom tree outputs
        f = int(classifier.get_full_region_complexity())
        e = int(classifier.get_error_region_complexity())
        c = int(classifier.get_correct_region_complexity())

        #print(f"{count},{accuracy},{f},{e},{c}")
        lst.append((count, accuracy))
        metric.correct = 0

    return lst

def get_size(n):
    with open(n) as file:
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
        data_size = len(data)
    return data_size

def get_patched(name, f, k):
    if len(name) == 1:
        data_file_path = name[0] + " pre.arff;" + name[0] + " post.arff";
        data_size = get_size(name[0] + " post.arff")
    else:
        data_file_path = name[0] + ".arff;" + name[1] + ".arff";
        data_size = get_size(name[1] + ".arff")

    data_file_paths = data_file_path.split(";")
    result = []
    sample_freq = k
    random_state = 0
    force_disable_patching=False
    force_enable_patching=f
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
        if prediction == actual_label:
            metric.correct += 1

        # train
        classifier.train()

        result = log_metrics(
            classifier_metrics_list[classifier_idx].instance_idx,
            sample_freq,
            metric,
            classifier,
            result, data_size)

    return result

def get_new(name, k):
    if len(name) == 1:
        data_file_path = name[0] + " pre.arff;" + name[0] + " post.arff";
        data_size = get_size(name[0] + " post.arff")
    else:
        data_file_path = name[0] + ".arff;" + name[1] + ".arff";
        data_size = get_size(name[1] + ".arff")


    data_file_paths = data_file_path.split(";")
    random_state = 0
    result = []
    sample_freq = k
    force_disable_patching=True
    force_enable_patching=False
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
        if prediction == actual_label:
            metric.correct += 1

        # train
        classifier.train()

        result = log_metrics(
            classifier_metrics_list[classifier_idx].instance_idx,
            sample_freq,
            metric,
            classifier,
            result, data_size)

    return result, data_size

def plt_result(name, k, f):
    acc_no_patch, data_size = get_new(name, k)
    acc_yes_patch = get_patched(name, f, k)
    acc_force_patch = get_patched(name, True, k)

    #print(acc_yes_patch)
    new_model_start = 0
    for i in range(1, len(acc_no_patch)):
        if acc_no_patch[i][0] == k:
            new_model_start = i
    #print(new_model_start)

    #if len(name)>1:
    new_model = acc_no_patch[new_model_start:]
    patched_model = acc_yes_patch[new_model_start:]
    forced_model = acc_force_patch[new_model_start:]
    
    fig = plt.figure(figsize=(6, 4))
    x_no_p = [i[0] for i in new_model]
    y_no_p = [i[1] for i in new_model]
    
    #plt.plot(x_no_p, y_no_p, color='C0', linestyle='-', label='ARF')
    sns.lineplot(x = x_no_p, y = y_no_p, color = "k", linestyle='-', label='Model without Transfer', alpha = 0.7)

    x_yes_p = [i[0] for i in patched_model]
    y_yes_p = [i[1] for i in patched_model]
    ax = sns.lineplot(x = x_yes_p, y = y_yes_p, color = "r", linestyle='-', label='Transferred Model', alpha = 0.7)
    #plt.plot(x_yes_p, y_yes_p, color='C1', linestyle='--', label='Patched Model')
    
    x_f_p = [i[0] for i in forced_model]
    y_f_p = [i[1] for i in forced_model]
    print(len(x_f_p))
    ax = sns.lineplot(x = x_f_p, y = y_f_p, color = "r", linestyle='--', label='Force Transferred Model', alpha = 0.7)
    

    sns.despine()
    sns.set_theme(style="ticks")
    #plt.title(name)
    plt.xlabel('No. instances', fontsize='20')
    plt.ylabel('Accuracy', fontsize='20')
    ax.set_xlim([0, data_size+200])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(.9, 0.15), fontsize=16)
    ax.get_legend().remove()
    if len(name) == 1:
        plt.savefig(name[0] +  '_v1 OPERA_' + str(k) + '.png', bbox_inches='tight')
    else:
        plt.savefig(name[0] + "_" + name[1] +  '_v1 OPERA_' + str(k) + '.png', bbox_inches='tight')
    plt.show()

name_lst = ['Arrowtown', 'Reefton', 'Masterton', 'Cromwell', 'Invercargill']
suffix = [" new temp", " new temp RH data"]
temp_names = []
temp_rh_names = []
for name in name_lst:
    temp_names.append(name+ suffix[0])
    temp_rh_names.append(name+ suffix[1])


f = False
k = 1000
# for name in name_lst:
#     plt_result([name], k , f)

# for name in temp_names:
#     plt_result([name], k , f)

# for name in temp_rh_names:
#     plt_result([name], k , f)

# name = [temp_names[-1]]
# plt_result(name, k, f)


k = 100
#for comb in itertools.permutations(name_lst, 2):
# for comb in itertools.permutations(temp_names, 2):
#     print(comb)
#     plt_result([comb[0], comb[1]], k, f)
#     print("")

# for comb in itertools.permutations(temp_rh_names, 2):
#     print(comb)
#     plt_result([comb[0], comb[1]], k, f)
#     print("")

plt_result([temp_names[-2], temp_names[-3]], k, f)
plt_result([temp_names[-1], temp_names[-3]], k, f)
#plt_result([temp_rh_names[-1], temp_rh_names[-2]], k, f)

# plt_result([temp_rh_names[-1], temp_rh_names[2]], k, f)