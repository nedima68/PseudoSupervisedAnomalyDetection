
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import savefig

def plot_2D_scatter(fig_title, train_data, test_data, test_labels, predictions, train_labels = [], column_names =  ['AE_RECONST','E_SCORES'], save_fig = False, show_fig = True, fig_file_name = 'Iforest_2D_scatter.png', 
                    show_train_samples = True, show_train_defect_samples = False):
    # default plot settings
    plt.rcParams['figure.dpi'] = 300
    #plt.rcParams['figure.figsize'] = [15, 10]
    plt.title(fig_title)

    if show_train_defect_samples:
        train_noise_indices = np.where(train_labels == -1)
        train_normal_indices = np.where(train_labels == 0)
    train_x = train_data[column_names[0]]
    train_y = train_data[column_names[1]]
    test_x = test_data[column_names[0]]
    test_y = test_data[column_names[1]]
    abnormal_indices = np.where(test_labels == 1)
    normal_indices = np.where(test_labels == 0)
    false_positives = [i  for i,n in enumerate(predictions) if (n==1 and i in normal_indices[0])]
    false_negatives = [i  for i,n in enumerate(predictions) if (n==0 and i in abnormal_indices[0])]
    true_negatives = [i  for i,n in enumerate(predictions) if (n==0 and i in normal_indices[0])]
    true_positives = [i  for i,n in enumerate(predictions) if (n==1 and i in abnormal_indices[0])]

    legend_graphs = []
    legend_labels = []
    
    if show_train_samples:
        if show_train_defect_samples:
            p11 = plt.scatter(train_x[train_normal_indices[0]], train_y[train_normal_indices[0]], c='white', s=20, edgecolor='k', alpha=0.8)
            legend_graphs.append(p11)
            legend_labels.append("normal training samples("+str(len(train_normal_indices[0])) + ')')

            p12 = plt.scatter(train_x[train_noise_indices[0]], train_y[train_noise_indices[0]], c='yellow', s=20, edgecolor='k', alpha=0.8)
            legend_graphs.append(p12)
            legend_labels.append("synthetic defects("+str(len(train_noise_indices[0])) + ')')

        else:
            p1 = plt.scatter(train_x, train_y, c='white', s=15, edgecolor='k', alpha=0.8)
            legend_graphs.append(p1)
            legend_labels.append("training observations")
    
    if len(true_negatives) > 0:
        p2 = plt.scatter(test_x[true_negatives], test_y[true_negatives], c='green', s=15, edgecolor='k')
        legend_graphs.append(p2)
        legend_labels.append("true negatives("+str(len(true_negatives)) + ')')

    if len(true_positives) > 0:
        p3 = plt.scatter(test_x[true_positives], test_y[true_positives], c='red', s=15, edgecolor='k')
        legend_graphs.append(p3)
        legend_labels.append("true positives("+str(len(true_positives)) + ')')

    if len(false_negatives) > 0:
        p4 = plt.scatter(test_x[false_negatives], test_y[false_negatives], c='blue', s=15, edgecolor='k')
        legend_graphs.append(p4)
        legend_labels.append("false negatives("+str(len(false_negatives)) + ')')

    if len(false_positives) > 0:

        p5 = plt.scatter(test_x[false_positives], test_y[false_positives], c='black', s=15, edgecolor='k')
        legend_graphs.append(p5)
        legend_labels.append("false positives("+str(len(false_positives)) + ')')
    
    
    plt.axis('tight')
    plt.xlabel(column_names[0])
    plt.ylabel(column_names[1])
    #plt.xlim((-2, 5))
    #plt.ylim((-2, 5))
    plt.legend(legend_graphs, legend_labels, loc="upper right")

    if save_fig:
        # saving the figure
        plt.savefig(fig_file_name, dpi=200)

    if show_fig:
        plt.show()
    plt.clf()
    plt.close()


def plot_multi_2D_scatter(fig_title, train_data, test_data, test_labels, predictions, train_labels = [], column_names =  ['SSIM', 'AE_RECONST','E_SCORES'], axis_combin = [('SSIM','E_SCORES'), ('AE_RECONST','E_SCORES')],
                          save_fig = False, show_fig = True, fig_file_name = 'Iforest_2D_multi_scatter.png', show_train_samples = True, show_train_defect_samples = False, full_axis_combination = True):
    #assert (len(column_names) == 3), "ERROR: number of column names must be 3"
    from itertools import combinations

    if show_train_defect_samples:
        train_noise_indices = np.where(train_labels == -1)
        train_normal_indices = np.where(train_labels == 0)
    abnormal_indices = np.where(test_labels == 1)
    normal_indices = np.where(test_labels == 0)
    false_positives = [i  for i,n in enumerate(predictions) if (n==1 and i in normal_indices[0])]
    false_negatives = [i  for i,n in enumerate(predictions) if (n==0 and i in abnormal_indices[0])]
    true_negatives = [i  for i,n in enumerate(predictions) if (n==0 and i in normal_indices[0])]
    true_positives = [i  for i,n in enumerate(predictions) if (n==1 and i in abnormal_indices[0])]

    if full_axis_combination:
        axis_combinations = list(combinations(column_names, 2))
    else:
        axis_combinations = axis_combin
    assert (len(axis_combinations) < 7), "ERROR: number of requested graphs must be smaller than 7"

    c_no = 0
    r_no = 0

    if len(axis_combinations) > 3:
        rows = 2
        cols = 3
    else:
        rows = 1
        cols = len(axis_combinations)
    #fig, ax = plt.subplots(1, 3, sharex='col', sharey='row')
    fig, ax = plt.subplots(rows, cols)
    plt.title(fig_title)
    for axes in axis_combinations:
        train_x = train_data[axes[0]]
        train_y = train_data[axes[1]] 
        test_x = test_data[axes[0]]
        test_y = test_data[axes[1]]

        #f = plt.figure(figsize=(15,5))
        #ax0 = f.add_subplot(131, projection='3d')
        if show_train_samples:
            if show_train_defect_samples:

                if len(axis_combinations) > 3:
                    ax[r_no, c_no].scatter(train_x[train_normal_indices[0]], train_y[train_normal_indices[0]], c='white', s=20, edgecolor='k', alpha=0.6, label="training(normal) data("+str(len(train_normal_indices[0]))+')')
                else:
                    ax[c_no].scatter(train_x[train_normal_indices[0]], train_y[train_normal_indices[0]], c='white', s=20, edgecolor='k', alpha=0.6, label="training(normal) data("+str(len(train_normal_indices[0]))+')')

                if len(axis_combinations) > 3:
                    ax[r_no, c_no].scatter(train_x[train_noise_indices[0]], train_y[train_noise_indices[0]], c='yellow', s=20, edgecolor='k', alpha=0.6, label="synthetic defects("+str(len(train_noise_indices[0]))+')')
                else:
                    ax[c_no].scatter(train_x[train_noise_indices[0]], train_y[train_noise_indices[0]], c='yellow', s=20, edgecolor='k', alpha=0.6, label="synthetic defects("+str(len(train_noise_indices[0]))+')')
            else:
                if len(axis_combinations) > 3:
                    ax[r_no, c_no].scatter(train_x, train_y, c='white', s=20, edgecolor='k', alpha=0.6, label="training data("+str(len(train_x))+')')
                else:
                    ax[c_no].scatter(train_x, train_y, c='white', s=20, edgecolor='k', alpha=0.6, label="training data("+str(len(train_x))+')')            

        if len(true_positives) > 0:
            if len(axis_combinations) > 3:
                ax[r_no, c_no].scatter(test_x[true_positives], test_y[true_positives], c='red', s=25, edgecolor='k', alpha=1.0, label="true positives("+str(len(true_positives))+')')
            else:
                ax[c_no].scatter(test_x[true_positives], test_y[true_positives], c='red', s=25, edgecolor='k', alpha=1.0, label="true positives("+str(len(true_positives))+')')

        if len(true_negatives) > 0:
            if len(axis_combinations) > 3:
                ax[r_no, c_no].scatter(test_x[true_negatives], test_y[true_negatives], c='green', s=25, edgecolor='k', alpha=1.0, label="true negatives("+str(len(true_negatives))+')')
            else:
                ax[c_no].scatter(test_x[true_negatives], test_y[true_negatives], c='green', s=25, edgecolor='k', alpha=1.0, label="true negatives("+str(len(true_negatives))+')')

        if len(false_negatives) > 0:
            if len(axis_combinations) > 3:
                ax[r_no, c_no].scatter(test_x[false_negatives], test_y[false_negatives], c='blue', s=30, edgecolor='k', alpha=1.0, label="false negatives("+str(len(false_negatives))+')')
            else:
                ax[c_no].scatter(test_x[false_negatives], test_y[false_negatives], c='blue', s=30, edgecolor='k', alpha=1.0, label="false negatives("+str(len(false_negatives))+')')

        if len(false_positives) > 0:
            if len(axis_combinations) > 3:
                ax[r_no, c_no].scatter(test_x[false_positives], test_y[false_positives], c='black', s=30, edgecolor='k', alpha=1.0, label="false positives("+str(len(false_positives))+')')
            else:
                ax[c_no].scatter(test_x[false_positives], test_y[false_positives], c='black', s=30, edgecolor='k', alpha=1.0, label="false positives("+str(len(false_positives))+')')

        if len(axis_combinations) > 3:
            ax[r_no, c_no].set(title = axes[0] + ' vs ' + axes[1] )
            ax[r_no, c_no].legend(loc = 1)
            ax[r_no, c_no].axis('tight')
            ax[r_no, c_no].set_xlabel(axes[0])
            ax[r_no, c_no].set_ylabel(axes[1])
        else:
            ax[c_no].set(title = axes[0] + ' vs ' + axes[1] )
            ax[c_no].legend(loc = 1)
            ax[c_no].axis('tight')
            ax[c_no].set_xlabel(axes[0])
            ax[c_no].set_ylabel(axes[1])
        c_no += 1
        if c_no > 2:
            c_no = 0
            r_no = 1
    
    if save_fig:
        # saving the figure
        plt.savefig(fig_file_name, dpi=300)

    if show_fig:
        plt.show()

    plt.clf()
    plt.close()


def plot_3D_scatter(fig_title, train_data, test_data, test_labels, predictions, column_names =  ['SSIM','AE_RECONST','E_SCORES','SVDD_SCORES'], axis_combin = ['SSIM','ISO_SCORES','SVDD_SCORES'], 
                    save_fig = False, show_fig = True, fig_file_name = 'Iforest_3D_scatter.png', show_train_samples = True, selected_axis_combination = True):

    if selected_axis_combination:
        train_x = train_data[axis_combin[0]]
        train_y = train_data[axis_combin[1]]
        train_z = train_data[axis_combin[2]]
        test_x = test_data[axis_combin[0]]
        test_y = test_data[axis_combin[1]]
        test_z = test_data[axis_combin[2]]
    else:
        train_x = train_data[column_names[0]]
        train_y = train_data[column_names[1]]
        train_z = train_data[column_names[2]]
        test_x = test_data[column_names[0]]
        test_y = test_data[column_names[1]]
        test_z = test_data[column_names[2]]

    abnormal_indices = np.where(test_labels == 1)
    normal_indices = np.where(test_labels == 0)
    false_positives = [i  for i,n in enumerate(predictions) if (n==1 and i in normal_indices[0])]
    false_negatives = [i  for i,n in enumerate(predictions) if (n==0 and i in abnormal_indices[0])]
    true_negatives = [i  for i,n in enumerate(predictions) if (n==0 and i in normal_indices[0])]
    true_positives = [i  for i,n in enumerate(predictions) if (n==1 and i in abnormal_indices[0])]


    f = plt.figure(figsize=(15,5))
    ax0 = f.add_subplot(111, projection='3d')
    if show_train_samples:
        ax0.scatter(train_x, train_y, train_z, c='white', s=15, edgecolor='k', alpha=0.8, label="training observations")
    
    if len(true_negatives) > 0:
        ax0.scatter(test_x[true_negatives], test_y[true_negatives], test_z[true_negatives], c='green', s=20, edgecolor='k', alpha=0.8, label="true negatives")

    if len(true_positives) > 0:
        ax0.scatter(test_x[true_positives], test_y[true_positives], test_z[true_positives], c='red', s=20, edgecolor='k', alpha=0.8, label="true positives")

    if len(false_negatives) > 0:
        ax0.scatter(test_x[false_negatives], test_y[false_negatives], test_z[false_negatives], c='blue', s=25, edgecolor='k', alpha=1.0, label="false negatives")

    if len(false_positives) > 0:
        ax0.scatter(test_x[false_positives], test_y[false_positives], test_z[false_positives], c='black', s=25, edgecolor='k', alpha=1.0, label="false positives")

    ax0.set(title = fig_title)
    ax0.set_xlabel(column_names[0])
    ax0.set_ylabel(column_names[1])
    ax0.set_zlabel(column_names[2])
    ax0.legend(loc = 1)
    if save_fig:
        # saving the figure
        plt.savefig(fig_file_name, dpi=300)

    if show_fig:
        plt.show()
    plt.clf()
    plt.close()