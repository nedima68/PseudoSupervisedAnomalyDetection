from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from utils.data_processing.StructuralSimilarityCalculator import calc_ssim
from sklearn.metrics import roc_auc_score
from skimage.measure import compare_ssim, compare_psnr
from skimage import data, img_as_float
#from skimage.metrics import structural_similarity 

import logging
import time
import torch
import torch.optim as optim
import numpy as np





class AutoEncoderTrainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        self.SSIM_train_scores = None
        self.AE_train_scores = None
        self.AE_test_scores = None
        self.AE_non_defect_reconst_score = None
        self.SSIM_test_scores = None
        self.non_defect_reconstrucion_measure = None

    def train(self, dataset: BaseADDataset, ae_net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        logger.info('Starting pretraining...')
        start_time = time.time()
        ae_net.train()
        ssim_scores = []
        AE_train_scores = []
        AE_train_latent_reps = []
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, _ = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = ae_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)
                loss.backward()
                optimizer.step()
                if (epoch == self.n_epochs - 1):
                    ssim_scores.extend(calc_ssim(inputs,outputs))
                    AE_train_scores.extend(scores.cpu().data.numpy().tolist())
                    AE_train_latent_reps.extend(ae_net.latent_rep.data.tolist())
                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))


        self.AE_train_time = time.time() - start_time
        logger.info('Pretraining time: %.3f' % self.AE_train_time)
        logger.info('SSIM scores are calculated...')
        logger.info('Finished pretraining.')
        self.SSIM_train_scores = ssim_scores
        self.AE_train_scores = AE_train_scores
        self.AE_train_latent_reps = AE_train_latent_reps
        # record a reconstruction measure for non_defect samples. This is a structural similarity score obtained by comparing
        # the original and reconstructed  images. For non-defect images this score shold be high. Defected images should have a smaller value
        # that most of the values here. So use a 5 percent quantile to compare during the test 
        self.non_defect_reconstrucion_measure = np.quantile(np.asarray(self.SSIM_train_scores), 0.02)
        logger.info('non_defect_reconstrucion_measure: %.3f' % self.non_defect_reconstrucion_measure)
        self.AE_non_defect_reconst_score = np.quantile(np.asarray(self.AE_train_scores), 0.99).item()
        logger.info('AE_non_defect_reconst_score: %.3f' % self.AE_non_defect_reconst_score)
        return ae_net

    def test(self, dataset: BaseADDataset, ae_net: BaseNet, use_train_set = False):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get test data loader
        if use_train_set:
            test_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        else:
            _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Testing autoencoder...')
        loss_epoch = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        test_ssim_scores = []
        ae_net.eval()
        AE_test_latent_reps = []
        with torch.no_grad():          
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = ae_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                #loss = torch.mean(scores)
                test_ssim_scores.extend(calc_ssim(inputs,outputs))
                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))
                AE_test_latent_reps.extend(ae_net.latent_rep.data.tolist())
                #loss_epoch += loss.item()
                #n_batches += 1

        #logger.info('Test set Loss: {:.8f}'.format(loss_epoch / n_batches))
        self.AE_test_latent_reps = AE_test_latent_reps
        _, labels, scores = zip(*idx_label_score)
        self.AE_test_scores = list(scores)
        labels = np.array(labels)
        scores = np.asarray(scores)
        self.AE_test_labels = labels
        self.SSIM_test_scores = test_ssim_scores       
        test_ssim_scores = np.asarray(test_ssim_scores)
        
        ssim_predictions = np.zeros_like(scores).astype(int)
        score_predictions = np.zeros_like(scores).astype(int)
        ssim_predictions[test_ssim_scores < self.non_defect_reconstrucion_measure] = 1
        score_predictions[scores > self.AE_non_defect_reconst_score] = 1
        self.AE_test_predictions = score_predictions
        
        if use_train_set:
            if len(np.argwhere(labels == -1)) > 0:          
                labels[labels == -1] = 1
            else:
                # there is no synthetic data so we are testing only non-defect samples (single label). 
                # we can't calculate AUC scores
                logger.info('Can not calculate AUC values due to lack of multiple labels (probably training samples without synthetic defects were provided)...')
                logger.info('(probably training samples without synthetic defects were provided)...')
                logger.info('Finished testing autoencoder.')
                return

        self.SSIM_pred_test_AUC_CONT = roc_auc_score(labels, test_ssim_scores)
        self.AE_Reconstr_AUC_CONT = roc_auc_score(labels, scores)
        self.SSIM_pred_test_AUC = roc_auc_score(labels, ssim_predictions)
        self.AE_Reconstr_AUC = roc_auc_score(labels, score_predictions)
        logger.info('Test set SSIM AUC (from predictions): {:.2f}%'.format(100. * self.SSIM_pred_test_AUC))
        logger.info('Test set AE_RECONS (cont.) AUC: {:.2f}%'.format(100. * self.AE_Reconstr_AUC_CONT))
        logger.info('Test set AE_RECONS (discrete) AUC: {:.2f}%'.format(100. * self.AE_Reconstr_AUC))
        
        self.AE_test_time = time.time() - start_time
        logger.info('Autoencoder testing time: %.3f' % self.AE_test_time)
        logger.info('Finished testing autoencoder.')


