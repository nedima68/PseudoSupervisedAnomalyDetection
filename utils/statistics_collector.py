import numpy as np
import math
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import sys

class MetricCollector():
    """
    This class collects a set of metrics during replications
      
    """
    def __init__(self, replication_num, metric_name_array, metric_collection_types = None, detailed_metric_assembly = False):
        """
        init function for the class

        Parameters
        ----------

        :arg replication_num: number of replications for a particlar experiment
        :arg metric_name_array: an array of strings (names) that specify names of the metrics. These must be valid "keys" in the "results" dictionary of the class implementing a  particular methodology (such as RobustAEImpl class implementing ROBUST_CAE)
        :arg metric_collection_types: an array of strings specifying the metric collection (or aggregation) method for a particular metric. 
                                      elements must be one of ('STRING_LIST', 'COUNT_MAX', 'MEAN_STD','MIN','MAX', 'MIN_MAX'). The size of the array must be the same as the size of metric_name_array
        :arg detailed_metric_assembly specifies whether metric_collection_types array is to be processed. If False metric_collection_types is ignored and metrics are aggregated in a pre-defined way
        
        """
        self.replication_num = replication_num
        self.metrics = metric_name_array
        self.metric_collection_types = metric_collection_types # can be a string array elements of which can be one of ('STRING_LIST', 'COUNT_MAX', 'MEAN_STD','MIN','MAX', 'MIN_MAX') 
        self.detailed_metric_assembly = detailed_metric_assembly
        self.replication_counter = 0
        self.metric_final_results = {}
        # initialize results array for each metric
        for metric in metric_name_array:
            self.metric_final_results[metric] = [] 

    def add_results(self, results):
        """
        This function adds a set of metrics to the collection

        Parameters
        ----------

        :arg results: a dictionary that holds all the results collected by a particular trainer. The names in the metric_name_array must exist in this dictionary
        
        """
        if self.replication_counter < self.replication_num:
            for metric in self.metrics:
                self.metric_final_results[metric].append(results[metric])

            self.replication_counter += 1
        else:
            raise Exception("The requested metric collection call of {}/{} exceeds the number of pre-defined replication".format(self.replication_counter, self.replication_num))

    def get_final_metrics(self, raw_data = False, convert_to_string = True):
        """
        This function returns the final metrics collected at the end of pre-defined number of replications

        Parameters
        ----------

        :arg raw_data: boolean if True return the full collected value array for each metric. If false mean and std_dev will be calculated and returned.
        :arg convert_to_string if True the mean and std values are converted to a single string int form "%{:.3f} ± {:.3f}"

        Returns
        ----------
        a dictionary conatining names and accumulated values of each metric. each value is either an array of values of mean and std of the collection
        """
        self.convert_to_string = convert_to_string
        assert self.replication_counter == self.replication_num, "ERROR: Didn't finish all replications. Please call this function in a proper place in your code ..."
        if raw_data:
            return self.metric_final_results

        if self.detailed_metric_assembly:
            return self.assemble_metrics_using_specific_methods()
        else:
            return self.assemble_standard_metrics()


    def assemble_metrics_using_specific_methods(self):
        for i, metric in enumerate(self.metrics):
            if (self.replication_num > 1):
                # metric_collection_type can be one of ('STRING_LIST','COUNT_MAX', 'MEAN_STD', 'AVG', 'MIN','MAX', 'MIN_MAX') 
                if self.metric_collection_types[i] == 'STRING_LIST':               
                    s = '('
                    for s_item in self.metric_final_results[metric]:
                        s += s_item + ','
                    t = list(s)
                    t[s.rfind(',')] = ')'
                    s = "".join(t)
                    self.metric_final_results[metric] = s
                elif self.metric_collection_types[i] == 'MEAN_STD':
                    mean = np.mean(np.array(self.metric_final_results[metric]))
                    std = np.std(np.array(self.metric_final_results[metric]))
                    if self.convert_to_string:
                        if metric.find('t_') == -1:
                            # this is an AUC metric
                            self.metric_final_results[metric] = "%{:.3f} \xb1 {:.2f}".format(100*mean, std*100)
                        else:
                            # this is a time metric
                            self.metric_final_results[metric] = "{:.3f} sec \xb1 {:.3f}".format(mean, std)
                    else:
                        self.metric_final_results[metric] = [mean, std]
                elif self.metric_collection_types[i] == 'COUNT_MAX':
                    max_occuring = self.find_max_occuring(self.metric_final_results[metric])
                    self.metric_final_results[metric] = max_occuring

                elif self.metric_collection_types[i] == 'AVG':
                    avg = np.mean(self.metric_final_results[metric])
                    self.metric_final_results[metric] = "{:.2f}".format(avg)

                elif self.metric_collection_types[i] == 'MIN':
                    min = np.min(self.metric_final_results[metric])
                    self.metric_final_results[metric] = str(min)

                elif self.metric_collection_types[i] == 'MAX':
                    max = np.max(self.metric_final_results[metric])
                    self.metric_final_results[metric] = str(max)

                elif self.metric_collection_types[i] == 'MIN_MAX':
                    min = np.min(self.metric_final_results[metric])
                    max = np.max(self.metric_final_results[metric])
                    self.metric_final_results[metric] = '[' + str(min) + ',' + str(max) + ']'

            elif self.replication_num == 1:
                if type(self.metric_final_results[metric][0]) == str:
                    self.metric_final_results[metric] = self.metric_final_results[metric][0]
                else:
                    if self.convert_to_string:
                        if metric.find('t_') != -1:
                            self.metric_final_results[metric] = "{:.3f} sec".format(self.metric_final_results[metric][0])                            
                        else:
                            self.metric_final_results[metric] = "{:.5f}".format(self.metric_final_results[metric][0])
                    else:
                        self.metric_final_results[metric] = self.metric_final_results[metric][0]
            else:
                raise Exception("Unexpected State: Replication number is zero")

        return self.metric_final_results

    def assemble_standard_metrics(self):
        for metric in self.metrics:
            if (self.replication_num > 1):
                if type(self.metric_final_results[metric][0]) == str:
                    s = '('
                    for s_item in self.metric_final_results[metric]:
                        s += s_item + ','
                    t = list(s)
                    t[s.rfind(',')] = ')'
                    s = "".join(t)
                    self.metric_final_results[metric] = s
                else:
                    mean = np.mean(np.array(self.metric_final_results[metric]))
                    std = np.std(np.array(self.metric_final_results[metric]))
                    if self.convert_to_string:
                        if metric.find('t_') == -1:
                            # this is an AUC metric
                            self.metric_final_results[metric] = "%{:.3f} \xb1 {:.2f}".format(100*mean, std*100)
                        else:
                            # this is a time metric
                            self.metric_final_results[metric] = "{:.3f} sec \xb1 {:.3f}".format(mean, std)
                    else:
                        self.metric_final_results[metric] = [mean, std]
            elif self.replication_num == 1:
                if type(self.metric_final_results[metric][0]) == str:
                    self.metric_final_results[metric] = self.metric_final_results[metric][0]
                else:
                    if self.convert_to_string:
                        if metric.find('t_') == -1:
                            self.metric_final_results[metric] = "%{:.3f}".format(100*self.metric_final_results[metric][0])
                        else:
                            self.metric_final_results[metric] = "{:.3f} sec".format(self.metric_final_results[metric][0])
                    else:
                        self.metric_final_results[metric] = self.metric_final_results[metric][0]
            else:
                raise Exception("Unexpected State: Replication number is zero")

        return self.metric_final_results

    def find_max_occuring(self, metric_list):
        item_dict = {}
        for m in metric_list:
            if m in item_dict.keys():
                item_dict[m] = item_dict[m] + 1
            else:
                item_dict[m] = 1

        max_index = np.argmax(np.array(list(item_dict.values())))
        max_occuring = list(item_dict.keys())[max_index]
        return max_occuring


def scale_remote_outliers(scores, R, dR):
    ''' 
    Some of the defect samples are very remote from the center
    This makes the scatter graph unreadable. This function scales down and squeezes the very remote
    samples to a sector which is between 10R and 5R in the scatter graph
    z(i) = [x(i) - min(x)] / [max(x) - min(x)] gives a value between 0 and 1 for values beyond 7R (which is min(x) value here)
    then 2R*z(i) + 2R places that values between 2R and 7R
    '''
    new_scores = scores.copy()
    max = np.amax(scores)
    if max > 7*R:
        big_indices = [i[0] for i in np.argwhere(scores > 7*R)]
        for idx in big_indices:
            new_scores[idx] = 2*R * ((scores[idx] - 7*R) / (max - 7*R)) + 2*R
        return new_scores
    else:
        return scores


def calc_running_std_mean(x):
  n = 0
  S = torch.tensor(np.zeros(x.shape))
  m = torch.tensor(np.zeros(x.shape))
  for x_i in x:
    n = n + 1
    m_prev = m
    m = m + (x_i - m) / n
    S = S + (x_i - m) * (x_i - m_prev)
  return {'mean': m, 'std_dev': torch.sqrt(S/n)}

def collect_statistics(results):
    flag = False
    statistics = {}
    last_scores = results['test_scores']
    inds, lbls, scores = zip(*last_scores)
    inds, lbls, scores = np.array(inds), np.array(lbls), np.array(scores)
    sample_size = len(scores)
    SSIM_scores = results['AE_SSIM_test_scores']   
    SSIM_reconst_measure = results['AE_SSIM_reconst_measure']
    AE_test_scores = results['AE_test_scores']
    AE_reconst_measure = results['AE_non_defect_reconst_measure']
    max_test_normals = np.amax(np.array([scores[i[0]] for i in np.argwhere(lbls == 0)]))
    max_train_normals = math.fabs(results['train_min_max_scores'][1])
    min_train_normals = math.fabs(results['train_min_max_scores'][0])

    statistics['AUC'] = results['test_auc']
    statistics['max_test_normals'] = max_test_normals
    statistics['max_train_normals'] = max_train_normals
    statistics['sample_size'] = sample_size

    R = results['HS_softB_radius']
    if results['objective'] == 'one-class':
        R = R
    else:
        R = R ** 2

    delta_R = 0
    statistics['R'] = R
    statistics['delta_R'] = delta_R

    scores = scores - delta_R
    scores = scale_remote_outliers(scores, R, delta_R)
    defects = zip(*[(i[0], scores[i[0]]) for i in np.argwhere(lbls == 1)])
    non_defects = zip(*[(i[0], scores[i[0]]) for i in np.argwhere(lbls == 0)])
    predictions = np.zeros_like(scores).astype(int)
    predictions[scores > 0] = 1
    SSIM_predictions = np.zeros_like(scores).astype(int)
    SSIM_predictions[np.array(SSIM_scores) < SSIM_reconst_measure] = 1
    AE_predictions = np.zeros_like(scores).astype(int)
    AE_predictions[np.array(AE_test_scores) > AE_reconst_measure] = 1
    non_confirm = np.where(predictions != SSIM_predictions)

    defect_idx = np.where(lbls == 1)
    non_defect_idx =  np.where(lbls == 0)
    FN_ = len(np.where(predictions[defect_idx] == 0)[0])
    FP_ = len(np.where(predictions[non_defect_idx] == 1)[0])
    sklearn_f1 = f1_score(lbls, predictions)
    sklearn_accuracy = accuracy_score(lbls, predictions)
    sklearn_precision = precision_score(lbls, predictions)
    sklearn_recall = recall_score(lbls, predictions)
    if flag:
        eps = R*0.35
        for i in range(len(scores)):
            if scores[i] < eps and scores[i] > -eps:
                if scores[i] < 0 and SSIM_predictions[i] == 1:
                    scores[i] = eps
                if scores[i] > 0 and SSIM_predictions[i] == 0:
                    scores[i] = - eps
        defects = zip(*[(i[0], scores[i[0]]) for i in np.argwhere(lbls == 1)])
        non_defects = zip(*[(i[0], scores[i[0]]) for i in np.argwhere(lbls == 0)])
        predictions = np.zeros_like(scores).astype(int)
        predictions[scores > 0] = 1
    statistics['adjusted_AUC'] = roc_auc_score(lbls, predictions)
    false_positives = []
    false_negatives = []
    FN_info = []
    FP_info = []
    sum_info = {}
    for i in range(len(scores)):
        if scores[i] > 0 and lbls[i] == 0:
            # find indices and scores which belong to samples that have a positive score (i.e beyond hypersphere radius R)   
            # but infact are non-defected (label == 0) (so these are wrongly classified as defected). 
            # positive here indicates defected sample
            false_positives.append((i, scores[i]))
            FP_info.append([('idx',i),('lbl', lbls[i]),('scr', predictions[i]),('ssim',SSIM_predictions[i]), ])
        elif scores[i] < 0 and lbls[i] == 1:
            # find indices and scores which belong to samples that have a negative score 
            # (i.e inside the hypersphere radius R),  
            # but infact are defected (label == 1)(so these are wrongly classified as normal). 
            # negative here indicates normal sample
            false_negatives.append((i, scores[i]))
            FN_info.append([('idx',i),('lbl', lbls[i]),('scr', predictions[i]),('ssim',SSIM_predictions[i]), ('AE',AE_predictions[i])])
    sum_info['FP_info'] = FP_info
    sum_info['FN_info'] = FN_info
    # Collect statistics
    statistics['false_positives'] = false_positives
    statistics['false_negatives'] = false_negatives
    if false_positives:
        _, FP_Scores = zip(*false_positives)
        statistics['FP_range'] = [np.min(np.array(FP_Scores)), np.max(np.array(FP_Scores))]

    else:
        statistics['FP_range'] = [0.0, 0.0]
    if false_negatives:
        _, FN_Scores = zip(*false_negatives)
        statistics['FN_range'] = [np.min(np.array(FN_Scores)), np.max(np.array(FN_Scores))]
        
    else:
        statistics['FN_range'] = [0.0,0.0]

    Precision_PPV = 0 # positive predictive value (PPV)
    Miss_rate_FNR = 0 #  false negative rate (FNR)
    Accuracy = 0
    F_score = 0
    Sensitivity_TPR = 0
    Specificity_TNR = 0
    try:
        FP = len(false_positives)
        FN = len(false_negatives)               
        P = len(np.argwhere(lbls == 1))
        N = len(np.argwhere(lbls == 0))
        TN = N - FP
        TP = P - FN
        Sensitivity_TPR = TP / (TP + FN) # also known as recall, probability of detection, True Positive Rate (TPR)
        Specificity_TNR = TN / (TN + FP) # also known as selectivity or true negative rate (TNR)
        # False Positive Rate (FPR) = 1 - Specificity or FP / (FP -TN)
        FPR = 1 - Specificity_TNR
        Precision_PPV = TP / (TP + FP) # positive predictive value (PPV)
        Miss_rate_FNR = FN / (FN + TP) #  false negative rate (FNR)
        Accuracy = (TP + TN) / (P + N)
        # The F-score can be used as a single measure of performance of the test for the positive class. 
        # The F-score is the harmonic mean of precision and recall:
        F_score = 2 * (Precision_PPV * Sensitivity_TPR) / (Precision_PPV + Sensitivity_TPR)
    except:
        print("Oops!",sys.exc_info()[0],"occured.")

    statistics['defects'] = defects
    statistics['non_defects'] = non_defects
    statistics['FP'] = FP
    statistics['FN'] = FN
    statistics['P'] = P
    statistics['N'] = N
    statistics['TN'] = TN
    statistics['TP'] = TP
    statistics['Sensitivity_TPR'] = Sensitivity_TPR
    statistics['recall'] = sklearn_recall
    statistics['Specificity_TNR'] = Specificity_TNR
    statistics['FPR'] = FPR
    statistics['Precision_PPV'] = Precision_PPV
    statistics['Miss_rate_FNR'] = Miss_rate_FNR
    statistics['Accuracy'] = Accuracy
    statistics['sklearn_accuracy'] = sklearn_accuracy
    statistics['F_score'] = F_score
    statistics['F1'] = sklearn_f1
    return statistics
