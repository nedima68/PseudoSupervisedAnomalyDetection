
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import json
import numpy as np
import matplotlib.pyplot as plt


def read_results(input_file):

    
    
    with open(input_file, 'r') as fd:
        results = json.load(fd)
    
    test_scores = results["RAE_test_scores"]
    outlier_radius = results["outlier_radius_ND"]
    labels, scores = zip(*test_scores)
    scores = np.array(scores)
    predictions = np.zeros_like(scores).astype(int)
    predictions[scores > outlier_radius] = 1
    return labels, predictions.tolist()


def plot_curves(fabric_codes, recall_set, precision_set, f1_score_set):
    from itertools import cycle
    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')

    for code, color in zip(fabric_codes, colors):
        l, = plt.plot(recall_set[code], precision_set[code], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for {0} (F1-Score = {1:0.2f})'
                      ''.format(code, f1_score_set[code]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curves and F1 scores of datasets')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

    plt.show()
    plt.close()

if __name__ == "__main__":
    fabric_codes = ['fabric_00', 'fabric_01', 'fabric_02','fabric_03', 'fabric_04',  'fabric_06']
    # For each dataset
    precisions = dict()
    recalls = dict()
    F1_scores = dict()
    f_name = "E:/Anomaly Detection/export/SM_VERBOSE_fabric-code_one-class_ROBUST_CAE_2.3_1.0_3_rd_128_results.json"
    for f_code in fabric_codes:
        input_file = f_name.replace("fabric-code",f_code)
        y_test, y_score = read_results(input_file)
        precisions[f_code], recalls[f_code], _ = precision_recall_curve(y_test, y_score)
        F1_scores[f_code] = f1_score(y_test, y_score)

    plot_curves(fabric_codes, precisions, recalls, F1_scores)
