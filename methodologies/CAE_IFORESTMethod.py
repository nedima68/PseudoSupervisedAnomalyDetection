import pyximport; pyximport.install()
import json
import torch
import logging
import numpy as np
import time
from base.base_dataset import BaseADDataset
from NeuralNetworks.main import build_network, build_autoencoder
from optimizers.deepSVDD_trainer import DeepSVDDTrainer
from optimizers.autoencoder_trainer import AutoEncoderTrainer
import pandas as pd
import matplotlib.pyplot as plt
from pylab import savefig
from sklearn.ensemble import IsolationForest
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from utils.visualization.plot_iforest_results import plot_2D_scatter, plot_3D_scatter, plot_multi_2D_scatter




class CAEIForestImpl(object):
    """A class for the Deep SVDD method.

    Attributes:
        objective: A string specifying the Deep SVDD objective (either 'one-class' or 'soft-boundary').
        nu: Deep SVDD hyperparameter nu (must be 0 < nu <= 1).
        R: Hypersphere radius R.
        c: Hypersphere center c.
        net_name: A string indicating the name of the neural network to use.
        net: The neural network \phi.
        ae_net: The autoencoder network corresponding to \phi for network weights pretraining.
        trainer: DeepSVDDTrainer to train a Deep SVDD model.
        optimizer_name: A string indicating the optimizer to use 
        ae_trainer: AutoEncoderTrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
        results: A dictionary to save the results.
    """

    def __init__(self,  variant: str = 'classic', nu: float = 0.1):
        """Inits DeepSVDD with one of the two objectives and hyperparameter nu."""

        assert variant in ('classic', 'extended'), "Objective must be either isolation FOrest i.e. 'ISOFOR' or Extended İsolation forest 'E_ISOFOR'."
        self.CAE_objective = 'one-class'
        self.variant = variant
        self.nu = nu
        self.R = 0.0  # hypersphere radius R
        self.c = None  # hypersphere center c

        self.net_name = None
        self.net = None  # neural network \phi

        self.trainer = None
        self.optimizer_name = None

        self.ae_net = None  # autoencoder network for pretraining
        self.ae_trainer = None
        self.ae_optimizer_name = None

              
        self.results = {
            'objective': self.variant,
            'train_time': None,
            'HS_center' : None,
            'HS_softB_radius': None,
            'train_min_max_scores': None,
            'train_loss_values': None,
            'train_sample_size': 0.0,
            'train_last_scores': None,
            'AE_SSIM_test_scores': None, # structral similarity comparison results of the test data
            'AE_test_scores': None, # Autoencoder reconstrcution error results of the test data
            'AE_SSIM_train_scores': None,  # structral similarity comparison results of the training data
            'AE_train_last_scores': None, # Autoencoder reconstrcution error results of the training  data
            'AE_SSIM_reconst_measure': None, # a treshold value to determine the anomalous samples based on the Structral similarity scores
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

    def fit_and_test_OC_SVM_RBF(self, create_graph = False):
        logger = logging.getLogger()
        OC_SVM_FullFeature = svm.OneClassSVM(nu=0.2, kernel="rbf", gamma="scale")
        OC_SVM_DualParam = svm.OneClassSVM(nu=0.2, kernel="rbf", gamma="scale")
        # prepare and load training data
        #TrainData_FullF = np.array(self.trainer.train_last_outputs)
        TrainData_FullF = np.array(self.ae_trainer.AE_train_latent_reps)
        TrainData_DualP = np.array(self.ae_trainer.AE_train_scores)
        TrainData_DualP = TrainData_DualP.reshape(-1, 1)
        # fit the models        
        OC_SVM_FullFeature.fit(TrainData_FullF)
        OC_SVM_DualParam.fit(TrainData_DualP)

        # prepare and Load test data for dual parameter SVM
        TestData_DualP = np.array(self.ae_trainer.AE_test_scores)
        TestData_DualP = TestData_DualP.reshape(-1,1)
        SVM_DP_test_Predictions = OC_SVM_DualParam.predict(TestData_DualP)
        SVM_DP_test_Predictions[SVM_DP_test_Predictions == 1] = 0
        SVM_DP_test_Predictions[SVM_DP_test_Predictions == -1] = 1
        # prepare and Load test data for full feature set SVM

        TestData_FullF = np.array(self.ae_trainer.AE_test_latent_reps)
        SVM_FullF_test_Predictions = OC_SVM_FullFeature.predict(TestData_FullF)
        SVM_FullF_test_Predictions[SVM_FullF_test_Predictions == 1] = 0
        SVM_FullF_test_Predictions[SVM_FullF_test_Predictions == -1] = 1
        _, labels, scores = zip(*self.trainer.test_scores)
        TestLabels = np.array(labels)
        SVM_FullF_AUC = roc_auc_score(TestLabels, SVM_FullF_test_Predictions) 
        SVM_SingP_AUC = roc_auc_score(TestLabels, SVM_DP_test_Predictions)

        logger.info('OC SVM with dual param data Test AUC : {:.2f}%'.format(100. * SVM_SingP_AUC))
        logger.info('OC SVM with full feature data Test AUC  : {:.2f}%'.format(100. * SVM_FullF_AUC))
        self.results['OC_SVM_RBF_AE_LREP_AUC'] = SVM_FullF_AUC
        self.results['OC_SVM_RBF_SINGP_AUC'] = SVM_SingP_AUC

    def fit_and_test_SKL_ISOF_AELatentRep(self, create_graph = False):
        logger = logging.getLogger()
        self.skl_isolation_forest = IsolationForest(n_estimators=500, max_samples=0.9, contamination = 'auto', max_features = 0.9, behaviour = 'new', random_state=np.random.RandomState(42))
        # Now get the full feature vector to create the actual training data
        TrainData = np.array(self.ae_trainer.AE_train_latent_reps)
        logger.info(TrainData.shape)
        
        # train the standard sklearn isolation forest implementation
        self.skl_isolation_forest.fit(TrainData)       
        #---------------------------------------------------------------------------------------------------------------------------------------
        # Testing the isolation forest trainer
        _, labels, scores = zip(*self.trainer.test_scores)
        # Testing the isolation forest trainer        
        TestData = np.array(self.ae_trainer.AE_test_latent_reps)       
        SKL_IF_test_Predictions = self.skl_isolation_forest.predict(TestData)       
        SKL_IF_test_Predictions[SKL_IF_test_Predictions  > 0] = 0
        SKL_IF_test_Predictions[SKL_IF_test_Predictions < 0] = 1
        # Get results
        num_defect =  labels.count(1)
        num_normal =  labels.count(0) 
        self.TestLabels = np.array(labels)
        sklearn_f1 = f1_score(self.TestLabels, SKL_IF_test_Predictions)
        sklearn_accuracy = accuracy_score(self.TestLabels, SKL_IF_test_Predictions)      
        sklearn_precision = precision_score(self.TestLabels, SKL_IF_test_Predictions)
        sklearn_recall = recall_score(self.TestLabels, SKL_IF_test_Predictions)
        SKL_AUC = roc_auc_score(self.TestLabels, SKL_IF_test_Predictions)
        
        self.results['test_defects_num'] = num_defect
        self.results['test_non_defect_num'] = num_normal
        self.results['predictions'] = self.trainer.predictions.tolist()
        logger.info('Sklearn AE latent rep Isolation Forest Test AUC : {:.2f}%'.format(100. * SKL_AUC))
        #logger.info('Extended full param Isolation Forest Test F1 score: {:.2f}%'.format(100. * sklearn_f1))
        self.results['SKL_ISO_AE_LREP_AUC'] = SKL_AUC
        
        logger.info('Finished AE latent rep. Sklearn Isolation Forest testing.')
        

    def fit_and_test_ISOF_single_param(self, create_graph = False):
        """
        Creates an aggregate parameter using autoencoder reconstruction error scores (AE_RECONST), SVDD network outlier distance score ( SVDD_SCORES) and structural similarity scores (SSIM)
        the aggregate parameter is defined as AGG = AE_RECONST * SVDD_SCORES / SSIM 
        this parameter is then used as a single parameter for training and testing  Isolation Forest 
        """
        logger = logging.getLogger()
        self.skl_isolation_forest = IsolationForest(max_samples='auto', contamination = 'auto', behaviour = 'new', random_state=np.random.RandomState(42))
        train_aggregate_params = np.array(self.ae_trainer.AE_train_scores)
        train_aggregate_params_md = train_aggregate_params.reshape((train_aggregate_params.shape[0],1))
        logger.info(train_aggregate_params.shape)
        IFTrainData = pd.DataFrame(train_aggregate_params, columns = ['aggregate'])       
        # train the standard sklearn isolation forest implementation
        self.skl_isolation_forest.fit(IFTrainData)
        # train the extended isolation forest implementation
        #---------------------------------------------------------------------------------------------------------------------------------------
        # Testing the isolation forest trainer
        _, labels, scores = zip(*self.trainer.test_scores)
        test_aggregate_params = np.array(self.ae_trainer.AE_test_scores)
        IFTestData = pd.DataFrame(test_aggregate_params, columns = ['aggregate'])
        test_aggregate_params_md = test_aggregate_params.reshape((test_aggregate_params.shape[0],1))
        SKL_IF_test_Predictions = self.skl_isolation_forest.predict(IFTestData)      
        SKL_IF_test_Predictions[SKL_IF_test_Predictions == 1] = 0
        SKL_IF_test_Predictions[SKL_IF_test_Predictions == -1] = 1
        # Get results
        num_defect =  labels.count(1)
        num_normal =  labels.count(0) 
        self.TestLabels = np.array(labels)
        sklearn_f1 = f1_score(self.TestLabels, SKL_IF_test_Predictions)
        sklearn_accuracy = accuracy_score(self.TestLabels, SKL_IF_test_Predictions)
        
        sklearn_precision = precision_score(self.TestLabels, SKL_IF_test_Predictions)
        sklearn_recall = recall_score(self.TestLabels, SKL_IF_test_Predictions)
        SKL_AUC = roc_auc_score(self.TestLabels, SKL_IF_test_Predictions)
        
        self.results['test_defects_num'] = num_defect
        self.results['test_non_defect_num'] = num_normal
        self.results['predictions'] = self.trainer.predictions.tolist()
        self.results['SKL_ISO_SINGP_AUC'] = SKL_AUC
       
        #logger.info('Sklearn single param Isolation Forest Test Accuracy: {:.2f}%'.format(100. * sklearn_accuracy))
        logger.info('Sklearn single param Isolation Forest Test AUC : {:.2f}%'.format(100. * SKL_AUC))       
        logger.info('Finished single param Isolation Forest testing.')
        if create_graph:
            plot_2D_scatter("IForest 2D Scatter Graph",self.IFTrainData, self.IFTestData, self.TestLabels, self.trainer.predictions, column_names =  ['AE_RECONST', 'E_SCORES'])
        

    def train(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0):
        """Trains the Deep SVDD model on the training data."""
        logger = logging.getLogger()
        #self.pretrain(dataset, optimizer_name, lr, 50, lr_milestones, batch_size, weight_decay, device, n_jobs_dataloader)
        self.optimizer_name = optimizer_name
        self.trainer = DeepSVDDTrainer(self.CAE_objective, self.R, self.c, self.nu, optimizer_name, lr=lr,
                                       n_epochs=n_epochs, lr_milestones=lr_milestones, batch_size=batch_size,
                                       weight_decay=weight_decay, device=device, n_jobs_dataloader=n_jobs_dataloader)
        # Get the model
        self.net = self.trainer.train(dataset, self.net)
        self.R = float(self.trainer.R.cpu().data.numpy())  # get float
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # get list
        self.results['t_train_SVDD'] = self.trainer.train_time
        self.results['HS_center'] = self.c
        self.results['HS_softB_radius'] = self.R
        self.results['train_min_max_scores'] = self.trainer.train_min_max_scores
        self.results['train_sample_size'] = len(dataset.train_set)
        self.results['train_loss_values'] = self.trainer.train_loss_values
        self.results['train_last_scores'] = self.trainer.train_last_scores
        

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the Deep SVDD model on the test data."""
        logger = logging.getLogger()
        self.ae_trainer.test(dataset, self.ae_net)
        self.results['AE_SSIM_test_scores'] = self.ae_trainer.SSIM_test_scores
        self.results['AE_test_scores'] = self.ae_trainer.AE_test_scores
        self.trainer.test(dataset, self.net)
        
        self.results['test_auc'] = self.trainer.test_AUC_CONT
        self.results['t_test_SVDD'] = self.trainer.test_time       
        self.results['test_scores'] = self.trainer.test_scores
        self.fit_and_test_ISOF_single_param(create_graph = False)
        self.fit_and_test_SKL_ISOF_AELatentRep(create_graph = False)
        self.fit_and_test_OC_SVM_RBF(create_graph = False)
        self.results['SSIM_AUC'] = self.ae_trainer.SSIM_pred_test_AUC
        self.results['AE_RECONS_AUC'] = self.ae_trainer.AE_Reconstr_AUC
        self.results['SVDD_AUC'] = self.trainer.pred_test_AUC
        self.results['SSIM_AUC_CONT'] = self.ae_trainer.SSIM_pred_test_AUC_CONT
        self.results['AE_RECONS_AUC_CONT'] = self.ae_trainer.AE_Reconstr_AUC_CONT
        self.results['SVDD_AUC_CONT'] = self.trainer.test_AUC_CONT
        


    def get_sample_test_scores(self, input_sample, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the Deep SVDD model on the test data."""

        if self.trainer is None:
            self.trainer = DeepSVDDTrainer(self.summary_writer, self.objective, self.R, self.c, self.nu,
                                           device=device, n_jobs_dataloader=n_jobs_dataloader)
        if self.net == None:
            logger = logging.getLogger()
            logger.error("trying to test an empty network. Please initialize and train the neural network first ... ")
            return
        
        scores, test_time = self.trainer.evaluate_sample(input_sample, self.net)
        return scores, test_time


    def pretrain(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        """Pretrains the weights for the Deep SVDD network \phi via autoencoder."""

        self.ae_net = build_autoencoder(self.net_name, self.NN_rep_dim)
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AutoEncoderTrainer(optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones,
                                    batch_size=batch_size, weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        self.ae_net = self.ae_trainer.train(dataset, self.ae_net)
        self.results['AE_SSIM_reconst_measure'] = self.ae_trainer.non_defect_reconstrucion_measure
        self.results['AE_SSIM_train_scores'] = self.ae_trainer.SSIM_train_scores
        self.results['AE_train_last_scores'] = self.ae_trainer.AE_train_scores
        self.results['t_train_CAE'] = self.ae_trainer.AE_train_time
        self.ae_trainer.test(dataset, self.ae_net)
        self.results['AE_SSIM_test_scores'] = self.ae_trainer.SSIM_test_scores
        self.results['AE_test_scores'] = self.ae_trainer.AE_test_scores
        self.results['AE_non_defect_reconst_measure'] = self.ae_trainer.AE_non_defect_reconst_score
        self.init_network_weights_from_pretraining()

    def init_network_weights_from_pretraining(self):
        """Initialize the Deep SVDD network weights from the encoder weights of the pretraining autoencoder."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)
        # Load the new state_dict
        self.net.load_state_dict(net_dict)

    def save_model(self, export_model, save_ae=True):
        """Save Deep SVDD model to export_model."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict() if save_ae else None

        torch.save({'R': self.R,
                    'c': self.c,
                    'train_min_max_scores': self.results['train_min_max_scores'],
                    'net_dict': net_dict,
                    'ae_net_dict': ae_net_dict}, export_model)

    def load_model(self, model_path, load_ae=False):
        """Load Deep SVDD model from model_path."""

        model_dict = torch.load(model_path)

        self.R = model_dict['R']
        self.c = model_dict['c']
        self.results['HS_center'] = self.c
        self.results['HS_softB_radius'] = self.R
        #self.results['train_min_max_scores'] = model_dict['train_min_max_scores']
        self.net.load_state_dict(model_dict['net_dict'])
        if load_ae:
            if self.ae_net is None:
                self.ae_net = build_autoencoder(self.net_name)
            self.ae_net.load_state_dict(model_dict['ae_net_dict'])

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)

    