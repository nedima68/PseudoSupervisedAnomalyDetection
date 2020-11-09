from utils.parse_json_files_to_latex_table import JSONtoLaTeXtableParser
import json


def convert_summary_results_to_latex_table(input_file, out_file, header_titles, data_keys):

    JSON_Latex_Converter = JSONtoLaTeXtableParser(out_file, header_titles)
    
    with open(input_file, 'r') as fd:
        results = json.load(fd)
    for key, data in results.items():
        JSON_Latex_Converter.add_line(data, data_keys)

    JSON_Latex_Converter.create_latex_table_2()
    JSON_Latex_Converter.write_table_to_file()
    print(JSON_Latex_Converter.table)

if __name__ == "__main__":

    dir_path = 'E:/Anomaly Detection/export/cumulative results'
    out_file_AITEX = dir_path + '/' + 'AITEX_results_latex_table.txt'
    out_file_MNIST = dir_path + '/' + 'MNIST_results_latex_table.txt'
    out_file_CIFAR = dir_path + '/' + 'CIFAR_results_latex_table.txt'
    input_result_file_AITEX = dir_path + '/' + 'AITEX_final_results_ROBUST_CAE_structured_nbatch[7,90]_rep.5_nr.[5]_a.[1.55]_Qntile995_AUCOptim.json'
    input_result_file_CIFAR = dir_path + '/' + 'CIFAR10_final_results-28-03-2020.json'
    input_result_file_MNIST = dir_path + '/' + 'MNIST_final_results-27-03-2020.json'
    #header_titles_AITEX = ['Dataset','Ft.rep. dim','DCAE-SSIM','DCAE-RCE','Deep SVDD','OC-SVM','Deep ISOF-FS'] # for comparative ISO results
    #header_titles_AITEX = ['Dataset','lambda','noise ratio', 'DCAE-SSIM','DCAE-RCE','Noisy Deep OC'] # for noisy Deep OC
    header_titles_AITEX = ['Dataset','alpha', 'noise ratio', 'noise type',' RCAE AUC ', 'F1 Score', 'Accuracy'] # for noisy Deep OC
    header_titles_MNIST_CIFAR = ['Normal Class','DCAE-SSIM','DCAE-RCE','Deep SVDD','OC-SVM','Deep ISOF-2P','Deep ISOF-FS']

    #data_keys_AITEX = ['dataset_code','NN_rep_dim',('summary_results', ['SSIM_AUC','AE_RECONS_AUC_CONT','SVDD_AUC_CONT','OC_SVM_RBF_FULLP_AUC','EXT_ISO_FULL_FEATURE_AUC_CONT'])]
    #data_keys_AITEX = ['dataset_code','lambda', 'noise_ratio', ('summary_results', ['SSIM_AUC','AE_RECONS_AUC_CONT','ISO_OPTIM_AUC_CONT'])]
    data_keys_AITEX = ['dataset_code','alpha', 'noise_ratio', 'noise_type', ('summary_results', ['RAE_RECONS_AUC_CONT', 'RAE_F1', 'RAE_ACCURACY'])]
    #data_keys_AITEX = ['dataset_code','alpha', 'noise_ratio', ('summary_results', ['RAE_RECONS_AUC_CONT', 'RAE_NOISE_SEP_AUC_CONT'])]
    data_keys_MNIST_CIFAR = ['dataset_code',('summary_results', ['SSIM_AUC','AE_RECONS_AUC_CONT','SVDD_AUC_CONT','OC_SVM_RBF_FULLP_AUC', 'EXT_ISO_TWOP_AUC_CONT','EXT_ISO_FULL_FEATURE_AUC_CONT'])]

    convert_summary_results_to_latex_table(input_result_file_AITEX, out_file_AITEX, header_titles_AITEX, data_keys_AITEX)