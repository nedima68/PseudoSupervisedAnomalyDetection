from click.testing import CliRunner
from AnomalyDetectionRobustAE import run_test
import os

import json

def RunAITEXFabricExperiments():
    """
    Executes multiple test runs for various anomaly detection methods
    """
    runner = CliRunner()
    cur_dir = os.getcwd()
    dataset_dir = 'AITEX_DS'
    online_dataset_source = [
                         cur_dir + os.sep + dataset_dir + os.sep + 'fabric_00',
                         cur_dir + os.sep + dataset_dir + os.sep + 'fabric_01',
                         cur_dir + os.sep + dataset_dir + os.sep + 'fabric_02',
                         cur_dir + os.sep + dataset_dir + os.sep + 'fabric_03',
                         cur_dir + os.sep + dataset_dir + os.sep + 'fabric_04',
                         cur_dir + os.sep + dataset_dir + os.sep + 'fabric_06'
                     ]

    dataset_names = ['custom', 'AITEX']    
    AITEX_patch_feeder_type = 'ONLINE'
    export_path = 'E:/Anomaly Detection/export'
    objectives = ['one-class']
    methodology = ['SVDD_ISO_SVM', 'ROBUST_CAE']
    nu_values = [0.1]
    NN_rep_dims = [256]
    device = 'cuda'
    seed = 5
    lambda_params = [1.6]
    noise_ratios = [3]
    noise_types = ['structured', 'statistical']
    lr_values = [0.01]
    n_epoch_values = [110]
    lr_milestone_1 = 80
    lr_milestone_2 = 110
    lr_milestone_3 = 250
    batch_sizes = [128]
    pretrain_values = [True, False]
    ae_n_epochs = [350]
    ae_lr_milestone_1 = 90
    ae_lr_milestone_2 = 150
    ae_lr_milestone_3 = 250
    net_names = ['DefectDetectCNN32x32']
    patch_size = '32x32'
    dataset_paths = online_dataset_source
    model_path = export_path +'/model.tar'
    final_results_path = export_path + '/AITEX_final_results.json'
    num_iter = len(net_names) * len(dataset_paths) * len(nu_values) * len(n_epoch_values) * len(batch_sizes) * len(NN_rep_dims)  * len(noise_ratios) * len(lambda_params)
    curr_iter = 1
    normal_class = 1
    run_replication_num = 5
    result_type = 'SUMMARY'
    cumulative_results = {}
    for net_name in net_names:
        for data in dataset_paths:
            for nu in nu_values:
                for epoch_num in n_epoch_values:
                    for batch_size in batch_sizes:
                        for objective in objectives:
                            for rep_dim in NN_rep_dims:
                                for lambda_param in lambda_params:
                                    for noise_ratio in noise_ratios:
                                        hyper_params = {}
                                        method = methodology[1]
                                        if AITEX_patch_feeder_type == 'ONLINE':
                                            dataset_code = data[data.find('fabric'):]
                                        else:
                                            dataset_code = data[data.find('fabric_0'):data.find('/'+patch_size)]
                                        parameter_list= [method, dataset_names[1], dataset_code, 
                                                            net_name, export_path, data, '--objective', objective, '--result_type', result_type, '--nu', nu, 
                                                            '--lambda_param', lambda_param, '--noise_ratio', noise_ratio, '--noise_type', noise_types[0], '--NN_rep_dim', rep_dim,
                                                            '--device', 'cuda', '--n_epochs', epoch_num, '--lr_milestone', lr_milestone_1, '--lr_milestone', lr_milestone_2, '--lr_milestone', lr_milestone_3, '--seed', seed, '--batch_size', 
                                                            batch_size, '--ae_n_epochs',ae_n_epochs[0], '--ae_lr_milestone', ae_lr_milestone_1, '--ae_lr_milestone', ae_lr_milestone_2, '--ae_lr_milestone', ae_lr_milestone_3,
                                                            '--normal_class', normal_class, '--replication_num', run_replication_num]
                                        result = runner.invoke(run_test, parameter_list)
                                        print('iter: {} / {}'.format(curr_iter, num_iter))                        
                                        print(result)
                                        print(result.output)       
                                        if method == 'ROBUST_CAE' or method == 'ROBUST_DEEP_SVDD' or method == 'SUDO_SUP_SVDD':
                                            json_res_file=export_path + '/' + result_type + '_'+ dataset_code + '_' + objective + '_' + method + '_' + str(lambda_param)+'_'+str(noise_ratio) +'_rd_' + str(rep_dim) + '_' + 'results.json'
                                                
                                                
                                        elif method == 'SVDD_IFOREST':
                                            json_res_file=export_path + '/' + result_type + '_'+ dataset_code + '_' + objective + '_' + method + '_' + str(nu)+'_'+str(epoch_num)+'_rd_' + str(rep_dim) + '_' + 'results.json'
                                        elif method == 'SVDD':    
                                            json_res_file=export_path + '/' + result_type + '_'+ dataset_code + '_' + objective + '_' + method + '_' + str(nu)+'_'+str(epoch_num)+'_'+'results.json'

                                        with open(json_res_file) as json_file:
                                            run_results = json.load(json_file)                               
                                                  
                                        cumulative_results[curr_iter] = run_results
                                        curr_iter += 1

                                        with open(final_results_path, 'w') as outfile:
                                            json.dump(cumulative_results, outfile)


    '''
    run_test(dataset_name, net_name, xp_path, data_path, load_config, load_model, objective, nu, device, seed,
         optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay, pretrain, ae_optimizer_name, ae_lr,
         ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay, n_jobs_dataloader, normal_class)
    result = runner.invoke(run_test, ['custom', 'E:/openCV', '--nu', '0.022', '--device', 'cuda', '--seed', '10'])
    #assert result.exit_code == 0
    '''

  
if __name__ == '__main__':
    RunAITEXFabricExperiments()


    

