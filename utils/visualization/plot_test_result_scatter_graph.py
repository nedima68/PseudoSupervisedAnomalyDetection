import json
import numpy as np
import matplotlib.pyplot as plt
import math
from utils.visualization.plot_images_grid import greatest_factor_pair

def plot_scatter_graph(title, defects, non_defects, false_negatives, false_positives, sample_size, R, delta_R, rows = 1):

    ax_ind = 0   
    fig, ax = plt.subplots(rows, sharex='col', sharey='row')
        
    mask = [1,1]
    if len(false_negatives) > 0:
        mask.append(1)
    else:
        mask.append(0)
    if len(false_positives) > 0:
        mask.append(1)
    else:
        mask.append(0)
        
    false_positives = zip(*false_positives)
    false_negatives = zip(*false_negatives)
    full_data = [defects, non_defects, false_negatives, false_positives]
    full_colors = ["red", "green", "blue", "black"]
    full_groups = ["defects", "non_defects", "false_negatives", "false_positives"]
    data = []
    colors = []
    groups = []
    for i, elem in enumerate(mask):
        if elem == 1:
            data.append(full_data[i])
            colors.append(full_colors[i])
            groups.append(full_groups[i])
                     
    #ax = fig.add_subplot(rows, 1, int(key), facecolor="1.0")
    sector_theta = 90
    for data, color, group in zip(data, colors, groups):
        a, r = data
        a = np.array(a)
        r = np.array(r) + R
        x = r * np.cos(np.radians(sector_theta * a / sample_size)) 
        y = r * np.sin(np.radians(sector_theta * a / sample_size)) 
        ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
        
    circle1 = plt.Circle((0, 0), R, color='b', fill=False, label="hypersphere")
    circle2 = plt.Circle((0, 0), R - delta_R, color='y', fill=False, label="hypersphere + delta")
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.set(title=title)
    ax.legend(loc = 1)        
    ax_ind += 1

    return fig, ax


class MultiGraphPlotter():
    def __init__(self, n_plots):
        self.ax_r_ind = 0
        self.ax_c_ind = 0
        self.cols, self.rows = greatest_factor_pair(n_plots)
        self.fig, self.ax = plt.subplots(self.rows, self.cols, sharex='col', sharey='row')

    def add_plot(self,title, defects, non_defects, false_negatives, false_positives, sample_size, R, delta_R):
        mask = [1,1]
        if len(false_negatives) > 0:
            mask.append(1)
        else:
            mask.append(0)
        if len(false_positives) > 0:
            mask.append(1)
        else:
            mask.append(0)
        
        false_positives = zip(*false_positives)
        false_negatives = zip(*false_negatives)
        full_data = [defects, non_defects, false_negatives, false_positives]
        full_colors = ["red", "green", "blue", "black"]
        full_groups = ["defects", "non_defects", "false_negatives", "false_positives"]
        data = []
        colors = []
        groups = []
        for i, elem in enumerate(mask):
            if elem == 1:
                data.append(full_data[i])
                colors.append(full_colors[i])
                groups.append(full_groups[i])
                     
        #ax = fig.add_subplot(rows, 1, int(key), facecolor="1.0")
        sector_theta = 90
        for data, color, group in zip(data, colors, groups):
            a, r = data
            a = np.array(a)
            r = np.array(r) + R
            x = r * np.cos(np.radians(sector_theta * a / sample_size)) 
            y = r * np.sin(np.radians(sector_theta * a / sample_size)) 
            #ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
            if (self.cols > 1):
                self.ax[self.ax_r_ind, self.ax_c_ind].scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
            else:
                self.ax[self.ax_r_ind].scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
        if (self.cols > 1):
            circle1 = plt.Circle((0, 0), R, color='b', fill=False, label="hyperphere")
            circle2 = plt.Circle((0, 0), R - delta_R, color='y', fill=False, label="hyperphere + delta")
            self.ax[self.ax_r_ind, self.ax_c_ind].add_artist(circle1)
            self.ax[self.ax_r_ind, self.ax_c_ind].add_artist(circle2)
            self.ax[self.ax_r_ind, self.ax_c_ind].set(title=title)
            self.ax[self.ax_r_ind, self.ax_c_ind].legend(loc = 1)
        else:
            circle1 = plt.Circle((0, 0), R, color='b', fill=False, label="hyperphere")
            circle2 = plt.Circle((0, 0), R - delta_R, color='y', fill=False, label="hyperphere + delta")
            self.ax[self.ax_r_ind].add_artist(circle1)
            self.ax[self.ax_r_ind].add_artist(circle2)
            self.ax[self.ax_r_ind].set(title=title)
            self.ax[self.ax_r_ind].legend(loc = 1)
                
        self.ax_r_ind += 1
        if self.ax_r_ind > self.rows-1 and self.cols > 1:
            self.ax_r_ind = 0
            self.ax_c_ind += 1
    

        return self.fig, self.ax

