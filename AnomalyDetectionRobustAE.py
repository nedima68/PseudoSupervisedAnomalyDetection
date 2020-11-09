
import click
import torch
import logging
import random
import json
import numpy as np
import pylab as plt
import cv2 as cv
import os
from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from datasets.preprocessing import normalize_data_bw_zero_one, global_contrast_normalization
from methodologies.CAE_IFORESTMethod import CAEIForestImpl
from methodologies.Robust_AE_Method import RobustAEImpl
from datasets.DatasetWrapper import load_dataset
#import cv2
from torchvision import transforms
from PIL import Image
from utils.visualization.plot_test_result_scatter_graph import plot_scatter_graph
from utils.visualization.create_heat_map import create_heat_map_for_img
from utils.statistics_collector import collect_statistics
from utils.statistics_collector import MetricCollector
from utils.visualization.display_sample_patches import generate_samples
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

fabric_code = 'fabric_06'
AITEX_path = os.getcwd() + os.sep + 'AITEX_DS' + os.sep + fabric_code
#AITEX_path = 'E:/Anomaly Detection/AITEX IMAGES/original_samples/fabric_06'
export_path = os.getcwd() + os.sep + 'export'
network = 'DefectDetectCNN32x32'
model_path = export_path +'/fabric_02_one-class_SVDD_model.tar'
methodology = 'ROBUST_CAE'
CIFAR10_path = 'E:/data'
MNIST_path = 'E:/data'
data_path = AITEX_path
replication_num = 1
normal_class = 1
ae_n_epochs = 20
ae_lr_milestone = (90,170, 250)
n_epochs = 3
rep_dim = 256
lr_milestone = (30,110, 250)
noise_type = 'structured'
################################################################################
# Settings
################################################################################
@click.command()
@click.argument('methodology', type=click.Choice(['SVDD_ISO_SVM', 'ROBUST_CAE']), default=methodology, required=True)
@click.argument('dataset_name', type=click.Choice(['AITEX', 'CIFAR10', 'MNIST']), default='AITEX', required=False)
@click.argument('fabric_code', type=str, default=fabric_code, required=True)
@click.argument('net_name', type=click.Choice(['CNN32x32Deep', 'DefectDetectCNN32x32', 'DefectDetectCNN64x64', 'MNIST_LeNet']), 
                default=network, required=False)
@click.argument('xp_path', type=click.Path(exists=True), default = export_path, required=False)
@click.argument('data_path', type=click.Path(exists=True),  default = data_path, required=False)
@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--objective', type=click.Choice(['one-class', 'soft-boundary']), default='one-class', help='Specify Deep SVDD objective ("one-class" or "soft-boundary").')
@click.option('--result_type', type=click.Choice(['VERBOSE', 'SUMMARY']), default='SUMMARY')
@click.option('--nu', type=float, default=0.01, help='Deep SVDD hyperparameter nu (must be 0 < nu <= 1).')
@click.option('--lambda_param', type=float, default=1.6, help='Robust CAE optimization hyperparameter (must be 0 < lambda_param).')
@click.option('--noise_ratio', type=int, default=3, help='ratio of the synthetically generated noise samples to be used in classifier training. Taken as a per cent.')
@click.option('--noise_type', type=click.Choice(['statistical', 'structured']), default=noise_type, help='type of the noise generated. If structured then a combination of perlin and julia_set geenrators are used')
@click.option('--NN_rep_dim', type=int, default=rep_dim, help='Size of the Neural Network Feature Vector (or latent/compressed representation dimension)')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=5, help='Set seed. If -1, use randomization.')
@click.option('--optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
              help='Name of the optimizer to use for Deep SVDD network training.')
@click.option('--lr', type=float, default=0.01,
              help='Initial learning rate for Deep SVDD network training. Default=0.001')
@click.option('--n_epochs', type=int, default=n_epochs, help='Number of epochs to train.')
@click.option('--lr_milestone', type=int, default=lr_milestone, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=1e-5,
              help='Weight decay (L2 penalty) hyperparameter for Deep SVDD objective.')
@click.option('--pretrain', type=bool, default=True,
              help='Pretrain neural network parameters via autoencoder.')
@click.option('--train', type=bool, default=True,
              help='train neural network parameters. If set to False only testing is performed.')
@click.option('--ae_optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
              help='Name of the optimizer to use for autoencoder pretraining.')
@click.option('--ae_lr', type=float, default=0.001,
              help='Initial learning rate for autoencoder pretraining. Default=0.001')
@click.option('--ae_n_epochs', type=int, default=ae_n_epochs, help='Number of epochs to train autoencoder.')
@click.option('--ae_lr_milestone', type=int, default=ae_lr_milestone, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--ae_batch_size', type=int, default=128, help='Batch size for mini-batch autoencoder training.')
@click.option('--ae_weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--normal_class', type=int, default=normal_class,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
@click.option('--replication_num', type=int, default=replication_num,
              help='Specify the number of run replication to get an average and a st_dev score for prediction AUC values.')
def run_test(methodology, dataset_name, fabric_code, net_name, xp_path, data_path, load_config, load_model, objective, 
             result_type, nu, lambda_param, noise_ratio, noise_type, nn_rep_dim, device, seed,
            optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay, pretrain, train, ae_optimizer_name, ae_lr,
            ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay, n_jobs_dataloader, normal_class, replication_num):
    """
    Deep SVDD, a fully deep method for anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """
    hyper_params = {}
    hyper_params['methodology'] = methodology
    hyper_params['dataset'] = dataset_name
    hyper_params['objective'] = objective
    hyper_params['dataset_code'] = fabric_code
    hyper_params['n_epochs'] = n_epochs
    hyper_params['ae_n_epochs'] = ae_n_epochs
    hyper_params['replications'] = replication_num
    hyper_params['nu'] = nu
    hyper_params['lambda'] = lambda_param
    hyper_params['noise_ratio'] = noise_ratio
    hyper_params['noise_type'] = noise_type
    hyper_params['NN_rep_dim'] = nn_rep_dim
    hyper_params['learn_rate'] = lr
    hyper_params['batch_size'] = batch_size
    # Get configuration
    cfg = Config(locals().copy())
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    if len(logger.handlers)  < 2:
        logger.addHandler(file_handler)

    # Print arguments
    logger.info('\n\nLog file is %s.' % log_file)
    logger.info('Data path is %s.' % data_path)
    logger.info('Export path is %s.' % xp_path)
    logger.info('Methodology is  %s.' % methodology)
    logger.info('Dataset: %s' % dataset_name)
    logger.info('Dataset Code: %s' % fabric_code)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Network: %s' % net_name)
    logger.info('NN rep dimension: %s' % nn_rep_dim)
    logger.info('Noise ratio: %s' % noise_ratio)
    logger.info('Noise type: %s' % noise_type)
    logger.info('lambda: %s' % lambda_param)
    logger.info('Total number of Replications: %s' % replication_num)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)
 
    # Set seed
    torch.backends.cudnn.deterministic = True
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)
    final_results = {}
    
   
    if methodology == 'SVDD_ISO_SVM':
        # Load data
        dataset = load_dataset(dataset_name, data_path, normal_class, dataset_code = fabric_code)
        # Print configuration
        logger.info('Isolation Forest variant: %s' % cfg.settings['objective'])
        variant = 'classic'

        metric_names = ['SSIM_AUC','SSIM_AUC_CONT','AE_RECONS_AUC', 'AE_RECONS_AUC_CONT', 'SVDD_AUC', 'SVDD_AUC_CONT',  'SKL_ISO_AE_LREP_AUC', 
                        'SKL_ISO_SINGP_AUC', 'OC_SVM_RBF_AE_LREP_AUC', 'OC_SVM_RBF_SINGP_AUC', 't_train_CAE','t_train_SVDD','t_test_SVDD']
                        

        metric_collection_types = ['MEAN_STD', 'MEAN_STD', 'MEAN_STD','MEAN_STD', 'MEAN_STD', 'MEAN_STD', 'MEAN_STD', 'MEAN_STD', 'MEAN_STD', 'MEAN_STD', 
                                   'MEAN_STD', 'MEAN_STD', 'MEAN_STD' ]
        metric_collector = MetricCollector(replication_num, metric_names)
        for i in range(replication_num):
            logger.info('Current replication no: %d' % i)
            # Initialize  model and set neural network \phi
            CAEIForest = CAEIForestImpl(variant)
            CAEIForest.set_network(net_name, nn_rep_dim)
            # If specified, load model
            if load_model:
                CAEIForest.load_model(model_path=load_model, load_ae=True) 
                logger.info('Loading SVDD model from %s.' % load_model)
            # Log pretraining details
            logger.info('Pretraining optimizer: %s' % cfg.settings['ae_optimizer_name'])
            logger.info('Pretraining learning rate: %g' % cfg.settings['ae_lr'])
            logger.info('Pretraining epochs: %d' % cfg.settings['ae_n_epochs'])
            logger.info('Pretraining learning rate scheduler milestones: %s' % (cfg.settings['ae_lr_milestone'],))
            logger.info('Pretraining batch size: %d' % cfg.settings['ae_batch_size'])
            logger.info('Pretraining weight decay: %g' % cfg.settings['ae_weight_decay'])

            # Pretrain model on dataset (via autoencoder)
            CAEIForest.pretrain(dataset,
                                optimizer_name=cfg.settings['ae_optimizer_name'],
                                lr=cfg.settings['ae_lr'],
                                n_epochs=cfg.settings['ae_n_epochs'],
                                lr_milestones=cfg.settings['ae_lr_milestone'],
                                batch_size=cfg.settings['ae_batch_size'],
                                weight_decay=cfg.settings['ae_weight_decay'],
                                device=device,
                                n_jobs_dataloader=n_jobs_dataloader)

 
            logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
            logger.info('Training learning rate: %g' % cfg.settings['lr'])
            logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
            logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
            logger.info('Training batch size: %d' % cfg.settings['batch_size'])
            logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

            # Train model on dataset
            CAEIForest.train(dataset,
                            optimizer_name=cfg.settings['optimizer_name'],
                            lr=cfg.settings['lr'],
                            n_epochs=cfg.settings['n_epochs'],
                            lr_milestones=cfg.settings['lr_milestone'],
                            batch_size=cfg.settings['batch_size'],
                            weight_decay=cfg.settings['weight_decay'],
                            device=device,
                            n_jobs_dataloader=n_jobs_dataloader)

            # Test model
            CAEIForest.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)
            logger.info('Adding summary results of dataset:{}/{} for replication num : {}/{}'.format(dataset_name,fabric_code,i+1,replication_num))
            metric_collector.add_results(CAEIForest.results)

         # Save results, model, and configuration
        CAEIForest.save_model(export_model=xp_path + '/' + fabric_code + '_' + objective + '_' + methodology + '_' + 'model.tar')
        cfg.save_config(export_json=xp_path + '/SVDD_IFOREST_config.json')
        if result_type == 'VERBOSE':
            CAEIForest.save_results(export_json=xp_path + '/' + result_type + '_'+ fabric_code + '_' + objective + '_' + methodology + '_' + str(nu)+'_'+str(n_epochs)+'_rd_' + str(nn_rep_dim) + '_' + 'results.json')                       
            final_results = CAEIForest.results
        elif result_type == 'SUMMARY':
            CAEIForest.save_results(export_json=xp_path + '/' + 'SM_VERBOSE' + '_'+ fabric_code + '_' + objective + '_' + methodology + '_' + str(nu)+'_'+str(n_epochs)+'_rd_' + str(nn_rep_dim) + '_' + 'results.json')
            summary_res = metric_collector.get_final_metrics(convert_to_string = True)
            hyper_params['summary_results'] = summary_res           
            export_json = xp_path + '/' + result_type + '_'+ fabric_code + '_' + objective + '_' + methodology + '_' + str(nu)+'_'+str(n_epochs)+'_rd_' + str(nn_rep_dim) + '_' + 'results.json'
            with open(export_json, 'w') as fp:
                json.dump(hyper_params, fp)
    
    

    elif methodology == 'ROBUST_CAE':
        # Load data

        #pretrain_dataset = load_dataset(dataset_name, data_path, normal_class, dataset_code = fabric_code)
        train_dataset = load_dataset(dataset_name, data_path, normal_class, gen_synthetic_defect = True, defect_ratio = noise_ratio, noise_type = noise_type, dataset_code = fabric_code)
        # Print configuration
        if noise_ratio > 0:
            metric_names = ['RAE_RECONS_AUC', 'RAE_RECONS_AUC_ND', 'RAE_RECONS_AUC_CONT','RAE_NOISE_SEP_AUC_CONT', 'RAE_NOISE_SEP_AUC_DISC', 'RAE_total_iterations', 'RAE_convergence', 
                            'false_positives', 'false_negatives', 'true_positives', 'true_negatives', 'RAE_F1', 'RAE_RECALL', 'RAE_ACCURACY','RAE_PRECISION','t_train_RAE', 't_test_RAE']
            metric_collection_types = ['MEAN_STD', 'MEAN_STD', 'MEAN_STD','MEAN_STD', 'MEAN_STD', 'AVG', 'COUNT_MAX', 'MIN_MAX', 'MIN_MAX', 'MIN_MAX', 'MIN_MAX', 'MEAN_STD', 'MEAN_STD', 'MEAN_STD', 'MEAN_STD', 'MEAN_STD', 'MEAN_STD']
        else:
            metric_names = ['RAE_RECONS_AUC', 'RAE_RECONS_AUC_ND', 'RAE_RECONS_AUC_CONT', 'RAE_total_iterations', 'RAE_convergence', 
                            'false_positives', 'false_negatives', 'true_positives', 'true_negatives', 'RAE_F1', 'RAE_RECALL', 'RAE_ACCURACY','RAE_PRECISION','t_train_RAE', 't_test_RAE']
            metric_collection_types = ['MEAN_STD', 'MEAN_STD', 'MEAN_STD', 'AVG', 'COUNT_MAX', 'MIN_MAX', 'MIN_MAX', 'MIN_MAX', 'MIN_MAX', 'MEAN_STD', 'MEAN_STD', 'MEAN_STD', 'MEAN_STD', 'MEAN_STD', 'MEAN_STD']

           

        metric_collector = MetricCollector(replication_num, metric_names, metric_collection_types = metric_collection_types, detailed_metric_assembly = True)
        for i in range(replication_num):
            logger.info('Current replication no: %d' % i)
            # Initialize DeepSVDD model and set neural network \phi
            RCAE = RobustAEImpl(xp_path, lambda_param, noise_ratio, verbose = True)
            RCAE.set_network(net_name, nn_rep_dim)
            # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
            if load_model:
                RCAE.load_model(model_path=load_model, load_ae=True) 
                logger.info('Loading Robust CAE model from %s.' % load_model)
            # Log pretraining details
            logger.info('training optimizer: %s' % cfg.settings['ae_optimizer_name'])
            logger.info('training learning rate: %g' % cfg.settings['ae_lr'])
            logger.info('training epochs: %d' % cfg.settings['ae_n_epochs'])
            logger.info('training learning rate scheduler milestones: %s' % (cfg.settings['ae_lr_milestone'],))
            logger.info('training batch size: %d' % cfg.settings['ae_batch_size'])
            logger.info('training weight decay: %g' % cfg.settings['ae_weight_decay'])

            # Pretrain model on dataset (via autoencoder)
            RCAE.train(train_dataset,
                                optimizer_name=cfg.settings['ae_optimizer_name'],
                                lr=cfg.settings['ae_lr'],
                                n_epochs=cfg.settings['ae_n_epochs'],
                                outer_iter=cfg.settings['n_epochs'],
                                lr_milestones=cfg.settings['ae_lr_milestone'],
                                batch_size=cfg.settings['ae_batch_size'],
                                weight_decay=cfg.settings['ae_weight_decay'],
                                device=device,
                                n_jobs_dataloader=n_jobs_dataloader)


            ## Test model
            RCAE.test(train_dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)
            logger.info('Adding summary results of dataset:{}/{} for replication num : {}/{}'.format(dataset_name,fabric_code,i+1,replication_num))
            metric_collector.add_results(RCAE.results)


         # Save results, model, and configuration
        RCAE.save_model(export_model=xp_path + '/' + fabric_code + '_' + objective + '_' + methodology + '_' + 'model.tar')
        cfg.save_config(export_json=xp_path + '/RCAE_config.json')
        if result_type == 'VERBOSE':
            RCAE.save_results(export_json=xp_path + '/' + result_type + '_'+ fabric_code + '_' + objective + '_' + methodology + '_'  + str(lambda_param)+'_'+str(noise_ratio) +'_rd_' + str(nn_rep_dim) + '_' + 'results.json')                       
            final_results = RCAE.results
        elif result_type == 'SUMMARY':
            RCAE.save_results(export_json=xp_path + '/' + 'SM_VERBOSE' + '_'+ fabric_code + '_' + objective + '_' + methodology + '_' + str(lambda_param)+'_'+str(noise_ratio) +'_rd_' + str(nn_rep_dim) + '_' + 'results.json')
            summary_res = metric_collector.get_final_metrics(convert_to_string = True)
            hyper_params['summary_results'] = summary_res           
            export_json = xp_path + '/' + result_type + '_'+ fabric_code + '_' + objective + '_' + methodology + '_'  + str(lambda_param)+'_'+str(noise_ratio) +'_rd_' + str(nn_rep_dim) + '_' + 'results.json'
            with open(export_json, 'w') as fp:
                json.dump(hyper_params, fp)

   

    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)


if __name__ == '__main__':
    run_test()
