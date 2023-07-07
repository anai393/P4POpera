from matplotlib import pyplot as plt
from matplotlib import animation
from skika.ensemble import adaptive_random_forest, pearl
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

class evaluate(object):
    #name = 'Arrowtown'
    #name ='Reefton'
    #name = 'Masterton'
    #name = 'Cromwell'
    name = ''

    step_size = 100

    def __init__(self, classifier):
        self.accuracy = 0
        self.num_instances = 0
        self.classifier = classifier

    
        self.sample_size = evaluate.step_size
        self.classifier.init_data_source(evaluate.name + " new.arff");

    def __call__(self):

        correct = 0

        sample_freq = self.sample_size

        for count in range(0, sample_freq):
            if not self.classifier.get_next_instance():
                break

            # test
            prediction = self.classifier.predict()

            actual_label = self.classifier.get_cur_instance_label()
            if prediction == actual_label:
                correct += 1

            # train
            self.classifier.train()

            self.classifier.delete_cur_instance()

        self.accuracy = correct / sample_freq
        self.num_instances += self.sample_size

        return self.num_instances, self.accuracy

class cls_evaluate(object):
    #name = 'Arrowtown'
    #name ='Reefton'
    #name = 'Masterton'
    #name = 'Cromwell'
    name = ''

    step_size = 100

    def __init__(self, classifier):
        self.accuracy = 0
        self.num_instances = 0
        self.classifier = classifier
        self.acc_lst = []

    
        self.sample_size = evaluate.step_size
        self.classifier.init_data_source(evaluate.name + " new.arff");

    def __call__(self):

        correct = 0

        sample_freq = self.sample_size

        for count in range(0, sample_freq):
            if not self.classifier.get_next_instance():
                break

            # test
            prediction = self.classifier.predict()

            actual_label = self.classifier.get_cur_instance_label()
            self.acc_lst.append((actual_label, prediction))
            if prediction == actual_label:
                correct += 1

            # train
            self.classifier.train()

            self.classifier.delete_cur_instance()

        self.accuracy = correct / sample_freq
        self.num_instances += self.sample_size

        return self.num_instances, self.acc_lst

def get_reults(name, step_size, s):
    print(name, step_size, s)
    evaluate.name = name
    evaluate.step_size = step_size

    print('completed evaluate.name = name')
    num_trees = 60
    max_num_candidate_trees = 120
    repo_size = 9000
    edit_distance_threshold = 90
    kappa_window = 50
    lossy_window_size = 100000000
    reuse_window_size = 0
    max_features = -1
    bg_kappa_threshold = 0
    cd_kappa_threshold = 0.4
    reuse_rate_upper_bound = 0.18
    warning_delta = 0.0001
    drift_delta = 0.00001
    enable_state_adaption = True
    enable_state_graph = True

    l = 1
    seed = s

    arf_classifier = adaptive_random_forest(num_trees,
                                            max_features,
                                            l,
                                            seed,
                                            warning_delta,
                                            drift_delta)
    ar = evaluate(arf_classifier)

    print('completed arf_classifier')
    pearl_classifier = pearl(num_trees,
                            max_num_candidate_trees,
                            repo_size,
                            edit_distance_threshold,
                            kappa_window,
                            lossy_window_size,
                            reuse_window_size,
                            max_features,
                            l,
                            seed,
                            bg_kappa_threshold,
                            cd_kappa_threshold,
                            reuse_rate_upper_bound,
                            warning_delta,
                            drift_delta,
                            enable_state_adaption,
                            enable_state_graph)

    print('completed pearl_classifier')
    pe = evaluate(pearl_classifier)

    print('completed evaluate')
    x_arf = []
    y_arf = []
    x_pearl = []
    y_pearl = []
    max_samples = 81

    if name == "Arrowtown":
        #n = 29800
        n = 10000
    elif name == "Reefton":
        n = 7309
    elif name == "Masterton":
        n = 4050
    elif name == "Cromwell":
        n = 7714
    else:
        n = 8135

    max_samples = n//evaluate.step_size

    print('before retrun')

    #29800 for Arrow
    #7309 for Reefton
    #4050 for Masterton
    #7714 for Cromwell
    #8135 for Invercargill
    return max_samples, ar, pe

def get_cls_results(name, step_size, s):
    evaluate.name = name
    evaluate.step_size = step_size

    num_trees = 60
    max_num_candidate_trees = 120
    repo_size = 9000
    edit_distance_threshold = 90
    kappa_window = 50
    lossy_window_size = 100000000
    reuse_window_size = 0
    max_features = -1
    bg_kappa_threshold = 0
    cd_kappa_threshold = 0.4
    reuse_rate_upper_bound = 0.18
    warning_delta = 0.0001
    drift_delta = 0.00001
    enable_state_adaption = True
    enable_state_graph = True

    l = 1
    seed = s

    arf_classifier = adaptive_random_forest(num_trees,
                                            max_features,
                                            l,
                                            seed,
                                            warning_delta,
                                            drift_delta)
    ar = cls_evaluate(arf_classifier)

    pearl_classifier = pearl(num_trees,
                            max_num_candidate_trees,
                            repo_size,
                            edit_distance_threshold,
                            kappa_window,
                            lossy_window_size,
                            reuse_window_size,
                            max_features,
                            l,
                            seed,
                            bg_kappa_threshold,
                            cd_kappa_threshold,
                            reuse_rate_upper_bound,
                            warning_delta,
                            drift_delta,
                            enable_state_adaption,
                            enable_state_graph)

    pe = cls_evaluate(pearl_classifier)

    x_arf = []
    y_arf = []
    x_pearl = []
    y_pearl = []
    max_samples = 81

    if name == "Arrowtown":
        #n = 29800
        n = 10000
    elif name == "Reefton":
        n = 7309
    elif name == "Masterton":
        n = 4050
    elif name == "Cromwell":
        n = 7714
    else:
        n = 8135

    max_samples = n//evaluate.step_size

    #29800 for Arrow
    #7309 for Reefton
    #4050 for Masterton
    #7714 for Cromwell
    #8135 for Invercargill
    return max_samples, ar, pe

def frames_arf(max_samples, arf):
    for i in range(max_samples):
        yield arf()

def frames_pearl(max_samples, pearl):
    for i in range(max_samples):
        yield pearl()

def plt_results(name_lst, step_size_lst):

    sns.set_theme(style="ticks", font_scale = 1.5)
    fig, axes = plt.subplots(5, 3, figsize=(16, 20), sharex='row', sharey='row')
    row = 0
    col = 0
    for step_size in step_size_lst:
        for name in name_lst:
            max_samples, ar, pe = get_reults(name, step_size)
            arf_results = []
            a_frame = frames_arf(max_samples, ar)
            for i in a_frame:
                arf_results.append(i)

            pearl_result = []
            b_frame = frames_pearl(max_samples, pe)
            for i in b_frame:
                pearl_result.append(i)

            x_arf = [i[0] for i in arf_results]
            y_arf = [i[1] for i in arf_results]
            #plt.plot(x_arf, y_arf, color='C0', linestyle='-', label='ARF')
            sns.lineplot(ax=axes[col, row], x = x_arf, y = y_arf, color = "k", linestyle='-', label='ARF')

            x_pearl = [i[0] for i in pearl_result]
            y_pearl = [i[1] for i in pearl_result]
            ax1 = sns.lineplot(ax=axes[col, row], x = x_pearl, y = y_pearl, color = "r", linestyle='--', label='PEARL')
            ax1.get_legend().remove()
            # if row != 2:
            #     ax1.get_legend().remove()
            col+=1
        row+=1
        col = 0

    #plt.title(evaluate.name)
    #plt.xlabel('no. instances')

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(.5, 0.92), fontsize=20)
    sns.despine()
    fig.text(0.5, 0.08, 'No. instances', ha='center', size = 24)
    fig.text(0.05, 0.5, "Accuracy", va='center', rotation='vertical', size = 24)
    plt.savefig('PEARL results.png', bbox_inches='tight')
    plt.show()

def plt_results2(name_lst, step_size_lst):
    sns.set_theme(style="ticks", font_scale = 1.5)
    for name in name_lst:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex='row', sharey='row')
        col = 0
        for step_size in step_size_lst:
            max_samples, ar, pe = get_reults(name, step_size, 1)
            arf_results = []
            a_frame = frames_arf(max_samples, ar)
            for i in a_frame:
                arf_results.append(i)

            pearl_result = []
            b_frame = frames_pearl(max_samples, pe)
            for i in b_frame:
                pearl_result.append(i)

            x_arf = [i[0] for i in arf_results]
            y_arf = [i[1] for i in arf_results]
            #plt.plot(x_arf, y_arf, color='C0', linestyle='-', label='ARF')
            sns.lineplot(ax=axes[col], x = x_arf, y = y_arf, color = "k", linestyle='-', label='ARF')

            x_pearl = [i[0] for i in pearl_result]
            y_pearl = [i[1] for i in pearl_result]
            ax1 = sns.lineplot(ax=axes[col], x = x_pearl, y = y_pearl, color = "r", linestyle='--', label='PEARL')
            ax1.set_title(r'$\alpha = $' + str(step_size), fontsize=20)
            ax1.get_legend().remove()


            col+=1
        #plt.title(evaluate.name)
        #plt.xlabel('no. instances')

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(.9, 0.15), fontsize=20)
        sns.despine()
        fig.text(0.515, -0.02, 'No. instances', ha='center', size = 24)
        fig.text(0.07, 0.5, "Accuracy", va='center', rotation='vertical', size = 24)
        #fig.text(0.515, 0.96, name, ha='center', size = 24)
        plt.savefig(name + ' PEARL.png', bbox_inches='tight')
        plt.show()

def get_avg(name_lst, step_size_lst):

    for step_size in step_size_lst:
        print("Step Size:", step_size)
        for name in name_lst:
            seed_lst = [0,1,2,3,4]
            arf_avg_lst = []
            pearl_avg_lst = []
            print("Name:", name)
            for seed in seed_lst:
                max_samples, ar, pe = get_reults(name, step_size, seed)
                arf_results = []
                a_frame = frames_arf(max_samples, ar)
                print("ARF:", a_frame)
                for i in a_frame:
                    arf_results.append(i)
                print('arf_results', arf_results)
                pearl_result = []
                print("PEARL:", pe)
                b_frame = frames_pearl(max_samples, pe)
                print("PEARL:", b_frame)
                for i in b_frame:
                    pearl_result.append(i)

                x_arf = [i[0] for i in arf_results]
                y_arf = [i[1] for i in arf_results]

                x_pearl = [i[0] for i in pearl_result]
                y_pearl = [i[1] for i in pearl_result]

                arf_avg = np.mean(y_arf)
                pearl_avg = np.mean(y_pearl)

                arf_avg_lst.append(arf_avg)
                pearl_avg_lst.append(pearl_avg)
            
            arf = np.mean(arf_avg_lst)
            arf_std = np.std(arf_avg_lst)
            print("ARF:", arf, arf_std)

            pearl = np.mean(pearl_avg_lst)
            pearl_std = np.std(pearl_avg_lst)
            print("PEARL:", pearl, pearl_std)

        print("")
  
def get_cls_acc(lst_lst):
    acc_cls_lst = []
    for lst in lst_lst:
        y_true = [i[0] for i in lst]
        y_pred = [i[1] for i in lst]
        matrix = confusion_matrix(y_true, y_pred)
        acc_cls = matrix.diagonal()/matrix.sum(axis=1)
        acc_cls_lst.append(list(acc_cls))
        #print(acc_cls)
    #print(acc_cls_lst)
    #print(len(acc_cls_lst))
    avg_acc_cls = np.mean(acc_cls_lst, axis = 0)
    return avg_acc_cls

def get_cls_prc_rc(lst_lst):
    prc_cls_lst = []
    rc_cls_lst = []
    for lst in lst_lst:
        y_true = [i[0] for i in lst]
        y_pred = [i[1] for i in lst]
        pr, rec, f, sup = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1, 2,3,4,5], zero_division = 0)
        prc_cls_lst.append(list(pr))
        rc_cls_lst.append(list(rec))
        #print(acc_cls)
    #print(acc_cls_lst)
    #print(len(acc_cls_lst))
    avg_prc_cls = np.mean(prc_cls_lst, axis = 0)
    avg_rc_cls = np.mean(rc_cls_lst, axis = 0)
    return avg_prc_cls, avg_rc_cls

def print_fmt(avg, std):
    print_lst = []
    for i in range(len(avg)):
        cur_avg = "{:.2f}".format(avg[i])
        cur_std = "{:.2f}".format(std[i])
        cur_p = cur_avg + u"\u00B1" + cur_std
        print_lst.append(cur_p)
    return print_lst
    
def get_cls_avg(name_lst, step_size_lst, dist_lst):
    for step_size in step_size_lst:
        print("Step Size:", step_size)
        for n in range(len(name_lst)):
            name = name_lst[n]
            dist = dist_lst[n]

            seed_lst = [0,1,2,3,4]
            #seed_lst = [0]
            arf_avg_lst_rc = []
            arf_avg_lst_prc = []
            pearl_avg_lst_rc = []
            pearl_avg_lst_prc = []
            print("Name:", name)
            for seed in seed_lst:
                max_samples, ar, pe = get_cls_results(name, step_size, seed)
                arf_results = []
                a_frame = frames_arf(max_samples, ar)
                for i in a_frame:
                    arf_results.append(i)

                pearl_result = []
                b_frame = frames_pearl(max_samples, pe)
                for i in b_frame:
                    pearl_result.append(i)

                y_arf = [i[1] for i in arf_results]

                y_pearl = [i[1] for i in pearl_result]
                #print(len(y_arf))

                arf_cls_avg_prc, arf_cls_avg_rc = get_cls_prc_rc(y_arf)
                pearl_cls_avg_prc, pearl_cls_avg_rc = get_cls_prc_rc(y_pearl)

                arf_avg_lst_prc.append(arf_cls_avg_prc)
                arf_avg_lst_rc.append(arf_cls_avg_rc)

                pearl_avg_lst_prc.append(pearl_cls_avg_prc)
                pearl_avg_lst_rc.append(pearl_cls_avg_rc)
            
            arf_prc = np.mean(arf_avg_lst_prc, axis=0)
            arf_prc_std = np.std(arf_avg_lst_prc, axis=0)
            arf_print_prc = print_fmt(arf_prc, arf_prc_std)
            print("ARF Precision:", arf_print_prc)

            arf_rc = np.mean(arf_avg_lst_rc, axis=0)
            arf_rc_std = np.std(arf_avg_lst_rc, axis=0)
            arf_print_rc = print_fmt(arf_rc, arf_rc_std)
            print("ARF Recall:", arf_print_rc)

            pearl_prc = np.mean(pearl_avg_lst_prc, axis=0)
            pearl_prc_std = np.std(pearl_avg_lst_prc, axis=0)
            pearl_print_prc = print_fmt(pearl_prc, pearl_prc_std)
            print("PEARL Precision:", pearl_print_prc)

            pearl_rc = np.mean(pearl_avg_lst_rc, axis=0)
            pearl_rc_std = np.std(pearl_avg_lst_rc, axis=0)
            pearl_print_rc = print_fmt(pearl_rc, pearl_rc_std)
            print("PEARL Recall:", pearl_print_rc)


            row_prc = []
            row_rc = []
            for i in range(len(arf_print_prc)):
                row_prc.append(dist[i])
                row_prc.append(arf_print_prc[i])
                row_prc.append(pearl_print_prc[i])
            for i in range(len(arf_print_rc)):
                row_rc.append(dist[i])
                row_rc.append(arf_print_rc[i])
                row_rc.append(pearl_print_rc[i])
            
            print("PRC row", row_prc)
            print("RC row", row_rc)


            
arrowtown = [14398, 6328, 3636, 5109, 310, 19]
reefton = [5122, 1377, 399, 381, 29, 1]
materton = [2153, 876, 318, 584, 103, 14]
cromwell = [3250, 1217, 702, 2035, 435, 75]
invercargill = [5398, 1670, 463, 576, 24, 4]
dist_lst = [arrowtown, reefton, materton, cromwell, invercargill]        
name_lst = ['Arrowtown', 'Reefton', 'Masterton', 'Cromwell', 'Invercargill']
step_size_lst = [10,100,1000]
#plt_results2(name_lst, step_size_lst)

get_avg(name_lst, step_size_lst)

# name_lst = ['Reefton']
# step_size_lst = [1000]
#get_cls_avg(name_lst, step_size_lst, dist_lst)