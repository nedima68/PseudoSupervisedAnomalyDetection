
import pyximport; pyximport.install()
import json
import torch
import logging
import numpy as np
import time
from base.base_dataset import BaseADDataset
from torch.utils.data import DataLoader
from NeuralNetworks.main import build_network, build_autoencoder
from optimizers.deepSVDD_trainer import DeepSVDDTrainer
from optimizers.robust_AE_trainer import RobustAETrainer
from base.base_dataset import CustomGenericImageFeeder
import pandas as pd
import matplotlib.pyplot as plt
from pylab import savefig
from torchvision import transforms
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from utils.visualization.plot_iforest_results import plot_2D_scatter, plot_3D_scatter, plot_multi_2D_scatter
from utils.data_processing.l21shrink import l21shrink
from base.base_data_types import PatchSize
from utils.visualization.display_sample_patches import plot_images_grid, tensor_to_img, save_img_patch
import GPUtil




class RobustAEImpl(object):
    """A class for the Robust Deep Convolutional Autoencoder (RDCAE) method.

    Attributes:


        net_name: A string indicating the name of the neural network to use.
        net: The neural network \phi.
        ae_net: The autoencoder network corresponding to \phi for network weights pretraining.
        optimizer_name: A string indicating the optimizer to use 
        ae_trainer: AutoEncoderTrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
        results: A dictionary to save the results.
    """

    def __init__(self,  export_path:str, lambda_param: float = 1.6, noise_ratio: int = 5, verbose = False):

        self.export_path = export_path
        self.lambda_param = lambda_param
        self.noise_ratio = noise_ratio
        self.net_name = None
        self.net = None  # neural network \phi
        self.verbose = verbose
        self.create_graph = True

        self.trainer = None
        self.optimizer_name = None
        self.error = 1.0e-5

        self.ae_net = None  # autoencoder network for pretraining
        self.ae_trainer = None
        self.ae_optimizer_name = None
        #rng = np.random.RandomState(42)
        self.epsilon = 0.0001
        
        self.results = {
            'train_time': None,
            'HS_center' : None,
            'HS_softB_radius': None,
            'train_min_max_scores': None,
            'train_loss_values': None,
            'train_sample_size': 0.0,
            'train_last_scores': None, 
            'AE_test_scores': None, # Autoencoder reconstrcution error results of the test data          
            'AE_train_last_scores': None, # Autoencoder reconstrcution error results of the training  data          
            'AE_non_defect_reconst_measure': None, # a treshold value to determine the anomalous samples based on the AE reconstruction error scores
            'test_auc': None,
            'test_time': None,
            'test_defects_num':  None,
            'test_non_defect_num':  None,
            'test_scores': None

        }
    
    def set_network(self, net_name, rep_dim):
        """Builds the neural network \phi."""
        self.net_name = net_name
        self.NN_rep_dim = rep_dim
        self.net = build_network(net_name, rep_dim)

    def train(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 80, outer_iter: int = 8,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):

        logger = logging.getLogger()
        self.ae_net = build_autoencoder(self.net_name, self.NN_rep_dim)
        self.ae_optimizer_name = optimizer_name
        train_epochs = outer_iter
        #AE_iteration = (n_epochs // train_epochs) 
        AE_n_epoch = n_epochs
        lr_update = (outer_iter // 3) * n_epochs if (outer_iter // 3) > 0 else 1
        logger.info("----------lr update : %d"% lr_update)
        total_it = 0
        self.ae_trainer = RobustAETrainer(optimizer_name, lr=lr, n_epochs=AE_n_epoch, lr_milestones=lr_milestones,
                                    batch_size=batch_size, weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
             
        images = dataset.train_set.get_all_images()
        labels = dataset.train_set.get_all_labels()

        X = images.to(device)
        #lamda = self.optim_lambda * X.shape[0]
        logger.info("lambda_param : {:.5f}".format(self.lambda_param))
        ## initialize L, S
        self.L = torch.zeros(X.shape).to(device)
        self.S = torch.zeros(X.shape).to(device)
        mu = (X.numel()) / (4.0 * X.norm(1)) 
        #shrink_param = (self.lambda_param / mu).item()
        self.shrink_param = self.lambda_param
        logger.info("mu parameter: {:.5f}".format(mu.item()))
        logger.info("shrink parameter: {:.5f}".format(self.shrink_param))
        LS0 = self.L + self.S

        XFnorm = X.norm('fro')

        logger.info("X shape: {}".format(X.shape))
        logger.info("L shape: {}".format(self.L.shape))
        logger.info("S shape: {}".format(self.S.shape))
        start_time = time.time()
        labels_tensor = torch.tensor(labels)
        convergence = "MAX_ITER"
        for it in range(train_epochs):          
            ## alternating project, first project to L
            self.L = X - self.S           
            self.ae_net = self.ae_trainer.train(self.L, labels, self.ae_net)
            ## get optmized L
            self.L = self.ae_net(self.L)
            ## alternating project, now project to S and shrink S
            self.S = l21shrink(self.shrink_param, (X - self.L), imsize= PatchSize(32,32), channels = 3, purpose = 'ANOMALY_DETECTION')
            self.S = torch.from_numpy(self.S).float().to(device)
            ## break criterion 1: the L and S are close enough to X
            c1 = (X - self.L - self.S).norm('fro') / XFnorm
            ## break criterion 2: there is no changes for L and S 
            c2 = torch.min(mu,torch.sqrt(mu)) * (LS0 - self.L - self.S).norm() / XFnorm

            logger.info("c1: {:.7f}".format(c1.item()))
            logger.info("c2: {:.7f}".format(c2.item()))

            ## save L + S for c2 check in the next iteration
            LS0 = self.L + self.S

            S_defect = self.S[labels_tensor == -1]
            S_non_defect = self.S[labels_tensor == 0]
            S_mean_non_def = S_non_defect.abs().sum(dim=1).mean() # take the sum of all elements of noise part, and than mean over all samples. Just to see progress
            S_mean_def = S_defect.abs().sum(dim=1).mean() # take the sum of all elements of noise part, and than mean over all samples. Just to see progress
            L_mean = self.L.sum(dim=1).mean() # take the sum of all elements of signal part, and than mean over all samples. Just to see progress
            logger.info("Outer iteration:{} mean-S_defect: {:.8f}, mean-S_non_defect: {:.8f}, mean-L: {:.5f} ".format(it, S_mean_def, S_mean_non_def, L_mean))

            total_it += AE_n_epoch
            logger.info("----------Total iter : %d"% total_it)
            logger.info("Testing for noise - signal separation convergence at iteration : %d"% total_it)
            self.ae_trainer.test_using_projections(X, labels, self.ae_net)
            #if self.ae_trainer.AE_Reconstr_AUC_CONT > 0.999 and S_mean_non_def  < self.error and S_mean_def > self.error * 100 or  (c1 < self.error * 10 and c2 < self.error * 10):
            if self.ae_trainer.AE_Reconstr_AUC_CONT > 0.999:
                logger.info("Signal and Noise separation has reached to a perfect point  .. so  VERY GOOD early break")
                convergence = "VERY_GOOD"
                break

            if S_mean_non_def  < self.error*0.1 and S_mean_def < self.error*100:
                logger.info("Signal and Noise separation NOT GOOD   .. so  PREMATURE early break")
                convergence = "EARLY_CONV"
                break

            if self.noise_ratio == 0:
                if S_mean_non_def  < self.error:
                    logger.info("S_mean_non_defect smaller than SIGNAL threshold...  and S_mean_def greater than DEFECT noise threshold ... so GOOD early break")
                    convergence = "GOOD"
                    break           
           

            if total_it % lr_update == 0:
                self.ae_trainer.lr = self.ae_trainer.lr / 10.0
                logger.info("----------Updating the learning rate to: {:.5f} ".format(self.ae_trainer.lr))

        #===========================================================================================
        RAE_train_time = time.time() - start_time
        GPUtil.showUtilization()
        logger.info('Trying to empty the GPU cache ...')
        torch.cuda.empty_cache()
        logger.info('GPU utilisation after emptying the GPU cache ...')
        GPUtil.showUtilization()
        self.outlier_radius = self.ae_trainer.outlier_radius
        self.results['AE_train_last_scores'] = self.ae_trainer.AE_train_scores
        self.results['AE_train_ND_last_scores'] = self.ae_trainer.AE_non_defect_train_scores
        self.results['t_train_RAE'] = RAE_train_time
        self.results['RAE_total_iterations'] = total_it
        self.results['RAE_convergence'] = convergence
        self.results['outlier_radius'] = self.outlier_radius
        self.results['train_sample_size'] = len(dataset.train_set)           
        self.results['outlier_radius_ND'] = self.ae_trainer.outlier_radius_ND  
        
        if self.verbose:
            import os
            dir_name = self.export_path + os.sep + 'images' +os.sep + 'RDCAE' + os.sep + dataset.dataset_code
            postfix_fname = '_a.' + str(self.lambda_param) + '-l.' + str(self.lambda_param) + '-nr.' + str(self.noise_ratio) + '.png'           
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
        if self.noise_ratio > 0:
            logger.info('Testing with X obtained from training  samples ...')
            self.ae_trainer.test_using_projections(X, labels, self.ae_net)
            #self.results['L_train_scores'] = self.ae_trainer.total_scores
            self.results['RAE_NOISE_SEP_AUC_CONT'] = self.ae_trainer.AE_Reconstr_AUC_CONT
            self.results['RAE_NOISE_SEP_AUC_DISC'] = self.ae_trainer.AE_Reconstr_AUC_ND
            #-----------------------------------------------------------------------------------------------------------------------------------------------------------
            if self.verbose:
                idx_sorted = np.argsort(np.array(self.ae_trainer.AE_test_scores))
                idx_defect_train = np.where(np.array(labels) == -1)
                idx_non_defect_train = np.where(np.array(labels) == 0)

                train_defect_X_imgs = X[idx_defect_train[0][:30]]
                train_non_defect_X_imgs = X[idx_non_defect_train[0][:30]]
                train_defect_S_imgs = self.S[idx_defect_train[0][:30]]
                train_defect_L_imgs = self.L[idx_defect_train[0][:30]]
                train_non_defect_S_imgs = self.S[idx_non_defect_train[0][:30]]
                train_non_defect_L_imgs = self.L[idx_non_defect_train[0][:30]]

                combined_S = torch.cat((train_non_defect_S_imgs, train_defect_S_imgs),0)
                combined_L = torch.cat((train_non_defect_L_imgs, train_defect_L_imgs),0)
                combined_orig = torch.cat((train_non_defect_X_imgs, train_defect_X_imgs),0)
                plot_images_grid(combined_S, export_img = dir_name + os.sep+ 'S_matrices_combined_training' + postfix_fname, title = 'S-Train [ND + SD]')
                plot_images_grid(combined_L, export_img = dir_name + os.sep+ 'L_matrices_combined_training' + postfix_fname, title = 'L-Train [ND + SD]')
                plot_images_grid(combined_orig, export_img = dir_name + os.sep+ 'X_matrices_combined_training' + postfix_fname, title = 'X-Train [ND + SD]')
                plot_images_grid(combined_S, export_img = dir_name + os.sep+ 'X(orig)_matrices_defect_training_data' + postfix_fname, title = 'Synthetic defects [Training samples]')
                plot_images_grid(train_non_defect_L_imgs, export_img = dir_name + os.sep+ 'L_matrices_non-defect_training_data' + postfix_fname, title = 'L [Non-defect training samples]')                
                plot_images_grid(train_non_defect_S_imgs, export_img = dir_name + os.sep+ 'S_matrices_non_defect_training_data' + postfix_fname, title = 'S(Noise) [Non-defect training samples]')
                plot_images_grid(train_defect_S_imgs, export_img = dir_name + os.sep+ 'S_matrices_defect_training_data' + postfix_fname, title = 'S(Noise) [Defect training samples]')
                plot_images_grid(train_defect_L_imgs, export_img = dir_name + os.sep+ 'L_matrices_defect_training_data' + postfix_fname, title = 'L [Defect training samples]')
        else:
            if self.verbose:
                idx_sorted = np.argsort(np.array(self.ae_trainer.AE_train_scores))           
                sorted_initial_L_imgs = self.L[idx_sorted[:40]]
                sorted_final_L_imgs = self.L[idx_sorted[-40:]]
                sorted_final_S_imgs = self.S[idx_sorted[-40:]]
                sorted_initial_S_imgs = self.S[idx_sorted[:40]]
                plot_images_grid(sorted_initial_L_imgs, export_img = dir_name + os.sep+ 'L_matrices_best_training_data' + postfix_fname, title = 'L matrices for best scores training data')
                plot_images_grid(sorted_final_L_imgs, export_img = dir_name + os.sep+ 'L_matrices_worst_training_data' + postfix_fname, title = 'L matrices for worst scores training data')
                plot_images_grid(sorted_initial_S_imgs, export_img = dir_name + os.sep+ 'S_matrices_predicted_best_training_data' + postfix_fname, title = ' S matrices for best scores training data')
                plot_images_grid(sorted_final_S_imgs, export_img = dir_name + os.sep+ 'S_matrices_predicted_worst_training_data' + postfix_fname, title = ' S matrices for worst scores training data')
            

        #===========================================================================================       

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the Deep SVDD model on the test data."""
        logger = logging.getLogger()
        images = dataset.test_set.get_all_images()
        labels = dataset.test_set.get_all_labels()
        X = images.to(device)
        
        train_labels = dataset.train_set.get_all_labels()
        no_defect_idx = np.where(np.array(train_labels) == 0)
        no_defect_L = self.L[no_defect_idx]
        L_mean = torch.mean(no_defect_L, dim=0)
        # get the reconstructed images using the trained AE
        R = self.ae_net(X)
        S = X - R
        save_img_patch(tensor_to_img(L_mean), prefix = "RDAE_L_mean_matrix")
        save_img_patch(tensor_to_img(no_defect_L[1]), prefix = "RDAE_no_defect_[1]")
        #L_test = X - S
        logger.info('Testing with actual test samples ...')
        self.ae_trainer.test_using_projections(X, labels, self.ae_net)      
        self.results['RAE_test_scores'] = self.ae_trainer.total_scores
        self.results['RAE_RECONS_AUC'] = self.ae_trainer.AE_Reconstr_AUC
        self.results['RAE_RECONS_AUC_CONT'] = self.ae_trainer.AE_Reconstr_AUC_CONT
        self.results['RAE_RECONS_AUC_ND'] = self.ae_trainer.AE_Reconstr_AUC_ND
        self.results['t_test_RAE'] = self.ae_trainer.RAE_test_time
        self.results['false_positives'] = self.ae_trainer.false_positives
        self.results['false_negatives'] = self.ae_trainer.false_negatives
        self.results['true_positives'] = self.ae_trainer.true_positives
        self.results['true_negatives'] = self.ae_trainer.true_negatives
        self.results['RAE_F1'] = self.ae_trainer.AE_F1_ND
        self.results['RAE_ACCURACY'] = self.ae_trainer.AE_accuracy_ND
        self.results['RAE_RECALL'] = self.ae_trainer.AE_recall_ND
        self.results['RAE_PRECISION'] = self.ae_trainer.AE_precision_ND
        #-----------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.verbose:
            dir_name = 'E:/temp/images/RDAE/' + dataset.dataset_code
            postfix_fname = '_a.' + str(self.lambda_param) + '-l.' + str(self.lambda_param) + '-nr.' + str(self.noise_ratio) + '.png'
            import os
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            idx_sorted = np.argsort(np.array(self.ae_trainer.AE_test_scores))        
            #sorted_initial10_L_imgs = L_test[idx_sorted[:60]]
            #sorted_final10_L_imgs = L_test[idx_sorted[-60:]]
            #sorted_final10_S_imgs = S[idx_sorted[-60:]]
            #sorted_initial10_S_imgs = S[idx_sorted[:60]]

            sorted_initial10_L_imgs = R[idx_sorted[:30]]
            sorted_final10_L_imgs = R[idx_sorted[-30:]]
            sorted_final10_S_imgs = S[idx_sorted[-30:]]
            sorted_initial10_S_imgs = S[idx_sorted[:30]]

            original_non_defect = X[idx_sorted[:30]]
            original_defect = X[idx_sorted[-30:]]
            combined_S = torch.cat((sorted_initial10_S_imgs, sorted_final10_S_imgs),0)
            combined_L = torch.cat((sorted_initial10_L_imgs, sorted_final10_L_imgs),0)
            combined_orig = torch.cat((original_non_defect, original_defect),0)
            plot_images_grid(combined_L, export_img = dir_name + os.sep + 'L_matrices_combined_Test' + postfix_fname, title = 'L-Test [ND + D]')
            plot_images_grid(combined_S, export_img = dir_name + os.sep + 'S_matrices_combined_Test' + postfix_fname, title = 'S_Test [ND + D]')
            plot_images_grid(combined_orig, export_img = dir_name + os.sep + 'original_matrices_combined_Test' + postfix_fname, title = 'X-Test [ND + D]')
            plot_images_grid(sorted_initial10_L_imgs, export_img = dir_name + os.sep + 'L_matrices_non-defect_test_data' + postfix_fname, title = 'L  [Non-defect test samples]')
            plot_images_grid(sorted_final10_L_imgs, export_img = dir_name + os.sep + 'L_matrices_defect_test_data' + postfix_fname, title = 'L [Defect test samples]')
            #plot_images_grid(sorted_final10_L_imgs, export_img = dir_name + os.sep + 'L_matrices_defect_test_data_NCUT' + postfix_fname, title = 'Images of NCUT Segmented Background L matrices for defect test data', apply_transforms = True)
            plot_images_grid(sorted_final10_S_imgs, export_img = dir_name + os.sep + 'S_matrices_defect_test_data' + postfix_fname, title = 'S(Noise) [Defect test samples]',  apply_transforms = False)
            plot_images_grid(sorted_initial10_S_imgs, export_img = dir_name + os.sep + 'S_matrices_non_defect_test_data' + postfix_fname, title = 'S(Noise) [Non-defect test samples]',  apply_transforms = False)
            #plot_images_grid(sorted_final10_S_imgs, export_img = dir_name + os.sep + 'S_matrices_defect_test_data_NCUT' + postfix_fname, title = 'Images of Noise S matrices for defect test data',  apply_transforms = True)
            plot_images_grid(original_non_defect, export_img = dir_name + os.sep + 'Original_lowscore_test_data' + postfix_fname, title = 'X (Input) [Non defect test data')
            plot_images_grid(original_defect, export_img = dir_name + os.sep + 'Original_highscore_test_data' + postfix_fname, title = ' X (Input) [Defect test samples]')


            TrainData = np.array([self.ae_trainer.SSIM_train_scores, self.ae_trainer.AE_train_scores])       
            Tr_TrainData = np.transpose(TrainData)
            RCAETrainData = pd.DataFrame(Tr_TrainData, columns = ['SSIM', 'AE_RECONST'])
            labels, scores = zip(*self.ae_trainer.total_scores)
            TestData = np.array([self.ae_trainer.SSIM_test_scores, self.ae_trainer.AE_test_scores])
            Tr_TestData = np.transpose(TestData)
            RCAETestData = pd.DataFrame(Tr_TestData, columns = ['SSIM', 'AE_RECONST'])

            if self.create_graph:
                plot_2D_scatter("Defect Detection Scatter Graph - " + dataset.dataset_code,RCAETrainData, RCAETestData, np.array(labels), self.ae_trainer.score_predictions_ND, train_labels = np.array(self.ae_trainer.AE_train_labels),  column_names =  ['SSIM','AE_RECONST'], 
                    save_fig = True, show_fig = False, fig_file_name = dir_name + os.sep + 'ScatterGraph' + postfix_fname, show_train_samples = True, show_train_defect_samples = True)
           

    def save_model(self, export_model, save_ae=True):
        """Save RDCAE model to export_model."""

        ae_net_dict = self.ae_net.state_dict() if save_ae else None
        torch.save({'R': self.outlier_radius,
                    'ae_net_dict': ae_net_dict}, export_model)

    def load_model(self, model_path, load_ae=False):
        """Load RDCAE model from model_path."""

        model_dict = torch.load(model_path)

        self.outlier_radius = model_dict['R']            
        self.ae_net.load_state_dict(model_dict['ae_net_dict'])

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)

    
