from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from base.base_dataset import CustomGenericImageFeeder
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils.data_processing.StructuralSimilarityCalculator import calc_ssim
from sklearn import metrics
from utils.data_processing.l21shrink import l21shrink
from base.base_data_types import PatchSize
#from skimage.metrics import structural_similarity 

import logging
import time
import torch
import torch.optim as optim
import numpy as np



class RobustAETrainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, noise_ratio: int  = 5, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

       
        self.AE_train_scores = None
        self.AE_test_scores = None
        self.AE_non_defect_reconst_score = None
        self.non_defect_reconstrucion_measure = None
        # Optimization parameters
        self.eps = 1e-6
        self.noise_ratio = noise_ratio

    def train(self, images, labels, ae_net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)
        #ae_net.reset_params()
        # Get train data loader
        dataset = CustomGenericImageFeeder(images, labels)
        train_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.n_jobs_dataloader)
        

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        logger.info('Starting pretraining...')
        start_time = time.time()
        ae_net.train()
        AE_train_scores = []
        AE_train_latent_reps = []
        label_score = []
        ssim_scores = []
        #inputs = dataset.clone()
        for epoch in range(self.n_epochs):
            
            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            epoch_start_time = time.time()  
            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                # Zero the network parameter gradients
                inputs, targets, _ = data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                # Update network parameters via backpropagation: forward + backward + optimize                
                outputs = ae_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)
                #----------------
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                n_batches += 1
                if (epoch == self.n_epochs - 1):
                    #AE_train_scores.extend(scores.cpu().data.numpy().tolist())
                    ssim_scores.extend(calc_ssim(inputs,outputs))
                    label_score += list(zip(targets.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))
                    
                    #AE_train_latent_reps.extend(ae_net.latent_rep.data.tolist())
           
            # log epoch statistics
            loss_epoch_batch = loss_epoch / n_batches
            #train_loss_values.append(loss_epoch_batch)
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch_batch))

        self.AE_train_time = time.time() - start_time
        logger.info('Pretraining time: %.3f' % self.AE_train_time)
        logger.info('Finished robust AE pretraining.')
        self.SSIM_train_scores = ssim_scores
        labels, scores = zip(*label_score)
        self.AE_train_scores = scores
        self.AE_train_labels = labels
        labels = np.array(labels)
        scores = np.array(scores)       
        non_defect_scores = [scores[i] for i in np.where(labels == 0)]
        synth_defect_scores = [scores[i] for i in np.where(labels == -1)]
        self.AE_non_defect_train_scores = non_defect_scores[0].tolist()

        test_labels = labels.copy()
        test_labels[ test_labels == -1] = 1
        num_syn_def = len(synth_defect_scores[0])
        mean_def = np.mean(np.asarray(synth_defect_scores[0]))
        # obtain a treshold using both non-defect samples and syntheticly generated defect samples. Noise ratio is used 
        # to define a reasonable quantile. (1 - noise_ratio / 100) is used to define the quantile ratio 
        self.outlier_radius = np.quantile(np.asarray(self.AE_train_scores), (1.0 - (self.noise_ratio - 1) / 100.0))
        # obtain a treshold using non-defect samples only. the value covering 99% of the non-defect sample scores is used as a treshold value get self.outlier_radius_ND 
        self.outlier_radius_ND = np.quantile(np.asarray(non_defect_scores), 0.999)        
        # record a reconstruction measure for non_defect samples. This is a similarity score obtained by comparing
        # the original and reconstructed  images. For non-defect images this score shold be low. Defected images should have a bigger value
        self.AE_non_defect_reconst_score = np.quantile(np.asarray(self.AE_train_scores), 0.99).item()
        logger.info('RAE_total_non_defect_reconst_score: %.3f' % self.AE_non_defect_reconst_score)
        logger.info('RAE outlier radius: %.3f' % self.outlier_radius)
        logger.info('RAE ND outlier radius: %.3f' % self.outlier_radius_ND)
        return ae_net

    def test(self, dataset: BaseADDataset, ae_net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Testing robust autoencoder...')
        loss_epoch = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        ae_net.eval()
        AE_test_latent_reps = []
        with torch.no_grad():          
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = ae_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)
                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))
                AE_test_latent_reps.extend(ae_net.latent_rep.data.tolist())
                loss_epoch += loss.item()
                n_batches += 1

        logger.info('Test set Loss: {:.8f}'.format(loss_epoch / n_batches))
        self.total_scores = idx_label_score
        self.AE_test_latent_reps = AE_test_latent_reps
        _, labels, scores = zip(*idx_label_score)
        self.AE_test_scores = list(scores)
        labels = np.array(labels)
        scores = np.asarray(scores)
        

        self.AE_Reconstr_AUC_CONT = roc_auc_score(labels, scores)
        score_predictions = np.zeros_like(scores).astype(int)
        score_predictions[scores > self.outlier_radius] = 1
        self.AE_Reconstr_AUC = roc_auc_score(labels, score_predictions)
        logger.info('Test set AE_RECONS (cont.) AUC: {:.2f}%'.format(100. * self.AE_Reconstr_AUC_CONT))
        logger.info('Test set AE_RECONS (discrete) AUC: {:.2f}%'.format(100. * self.AE_Reconstr_AUC))
        
        self.RAE_test_time = time.time() - start_time
        logger.info('Autoencoder testing time: %.3f' % self.RAE_test_time)
        logger.info('Finished testing autoencoder.')

    def test_using_projections(self, images, labels, ae_net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Testing
        logger.info('Testing robust autoencoder...')
        #loss_epoch = 0.0
        #n_batches = 0
        start_time = time.time()
        idx_label_score = []
        ae_net.eval()
        test_ssim_scores = []
        #AE_test_latent_reps = []
        with torch.no_grad():          
            
            inputs = images
            inputs = inputs.to(self.device)
            outputs = ae_net(inputs)
            scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
            #loss = torch.mean(scores)
            # Save triple of (idx, label, score) in a list
            idx_label_score += list(zip(labels,
                                        scores.cpu().data.numpy().tolist()))
            test_ssim_scores.extend(calc_ssim(inputs,outputs))

        #logger.info('Test set Loss: {:.8f}'.format(loss_epoch / n_batches))
        self.total_scores = idx_label_score
        #self.AE_test_latent_reps = AE_test_latent_reps
        self.SSIM_test_scores = test_ssim_scores  
        labels, scores = zip(*idx_label_score)
        self.AE_test_scores = list(scores)
        labels = np.array(labels)
        scores = np.array(scores)
        test_labels = labels.copy()
        test_labels[ test_labels == -1] = 1
        self.AE_Reconstr_AUC_CONT = roc_auc_score(test_labels, scores)
        score_predictions_Combined = np.zeros_like(scores).astype(int)
        score_predictions_Combined[scores > self.outlier_radius] = 1

        self.score_predictions_ND = np.zeros_like(scores).astype(int)
        self.score_predictions_ND[scores > self.outlier_radius_ND] = 1

        abnormal_indices = np.where(labels == 1)
        normal_indices = np.where(labels == 0)
        self.false_positives = len([i  for i,n in enumerate(self.score_predictions_ND) if (n==1 and i in normal_indices[0])])
        self.false_negatives = len([i  for i,n in enumerate(self.score_predictions_ND) if (n==0 and i in abnormal_indices[0])])
        self.true_negatives = len([i  for i,n in enumerate(self.score_predictions_ND) if (n==0 and i in normal_indices[0])])
        self.true_positives = len([i  for i,n in enumerate(self.score_predictions_ND) if (n==1 and i in abnormal_indices[0])])

        score_predictions_Approx = np.zeros_like(scores).astype(int)
        score_predictions_Approx[scores > self.AE_non_defect_reconst_score] = 1

        self.AE_Reconstr_AUC = roc_auc_score(test_labels, score_predictions_Combined)
        self.AE_Reconstr_AUC_ND = roc_auc_score(test_labels, self.score_predictions_ND)
        self.AE_F1_ND = f1_score(test_labels, self.score_predictions_ND)
        self.AE_accuracy_ND = accuracy_score(test_labels, self.score_predictions_ND)
        self.AE_recall_ND = recall_score(test_labels, self.score_predictions_ND)
        self.AE_precision_ND = precision_score(test_labels, self.score_predictions_ND)

        self.AE_Reconstr_AUC_Approx = roc_auc_score(test_labels, score_predictions_Approx)
        logger.info('Projected  AE_RECONS (cont.) AUC: {:.2f}%'.format(100. * self.AE_Reconstr_AUC_CONT))
        logger.info('Projected  AE_RECONS (discrete) AUC: {:.2f}%'.format(100. * self.AE_Reconstr_AUC))
        logger.info('Projected  AE_RECONS ND Based (discrete) AUC: {:.2f}%'.format(100. * self.AE_Reconstr_AUC_ND))
        logger.info('Projected  AE_RECONS Approx(0.98) (discrete) AUC: {:.2f}%'.format(100. * self.AE_Reconstr_AUC_Approx))
        logger.info('False positives: {}'.format(self.false_positives))
        logger.info('False negatives: {}'.format(self.false_negatives))
        
        self.RAE_test_time = time.time() - start_time
        logger.info('Robust Autoencoder testing time: %.3f' % self.RAE_test_time)
        logger.info('Finished testing robust autoencoder.')


