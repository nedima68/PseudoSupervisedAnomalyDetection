from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from datasets.preprocessing import global_contrast_normalization
from sklearn.metrics import roc_auc_score

import pyximport; pyximport.install()
import pandas as pd
from sklearn.ensemble import IsolationForest

import logging
import time
import torch
import torch.optim as optim
import numpy as np


class DeepSVDDTrainer(BaseTrainer):

    def __init__(self, objective, R, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.train_pix_std_dev = None
        self.train_pix_mean = None
        self.nu = nu

        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated
        #self.summary_writer = tensorboard_writer
        # Results
        self.train_time = None
        self.train_loss_values = None
        self.train_min_max_scores = None
        self.train_last_scores = None
        self.test_AUC_CONT = None
        self.test_time = None
        self.test_scores = None

    def train(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        train_loss_values = []
        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # initialize image stats
        #if self.train_pix_mean is None or self.train_pix_std_dev is None:
        #    logger.info('Initializing pixelwise normal (non-defect) sample statistics.')
        #    self.init_train_img_stats(train_loader)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()

        self.train_last_scores = []
        self.train_last_outputs = []
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
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                    if (epoch == self.n_epochs - 1):
                        #self.train_last_scores.append(scores.cpu().detach().numpy())
                        self.train_last_scores.extend(scores.data.tolist())
                        self.train_last_outputs.extend(outputs.data.tolist())
                       
                else:
                    loss = torch.mean(dist)
                    if (epoch == self.n_epochs - 1):
                        self.train_last_scores.extend(dist.data.tolist())
                        self.train_last_outputs.extend(outputs.data.tolist())
                    
                loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            loss_epoch_batch = loss_epoch / n_batches
            train_loss_values.append(loss_epoch_batch)
            epoch_train_time = time.time() - epoch_start_time
            
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch_batch))

        self.train_time = time.time() - start_time
        self.train_loss_values  = train_loss_values
        max_score = float(np.amax(np.array(self.train_last_scores)))
        min_score = float(np.amin(self.train_last_scores))
        self.train_min_max_scores = [min_score, max_score]
        if self.objective == 'one-class':
            logger.info("Trying to derive a suitable R value from training scores ... ")
            mean = np.mean(np.array(self.train_last_scores))
            std = np.std(np.array(self.train_last_scores))
            if std < 0.15:
                # distributionis too narrow, this would probably result in many false positives                
                # so we opt for a larger R value
                logger.info("Regenerating a suitable distribution from training scores ... ")
                new_dist = np.random.normal(mean + mean * 0.2, std * 2.2, len(self.train_last_scores))
                self.R = torch.tensor(np.amax(np.array(new_dist)))
                logger.info('R value: %.3f' % self.R.item())
            else:
                # new_dist = np.random.normal(mean + mean * 0.1, std, len(self.train_last_scores))
                # self.R = torch.tensor(np.amax(np.array(new_dist)))
                self.R = torch.tensor(max_score)
                logger.info('R value: %.3f' % self.R.item())
            #TO DO: trying ... remove if not OK
            self.R = torch.tensor(np.quantile(np.asarray(self.train_last_scores), 0.98))
            logger.info('quantile based R value: %.3f' % self.R.item())

        logger.info('Training time: %.3f' % self.train_time)
        logger.info('Training max score: %.3f' % max_score)
        logger.info('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet): 
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        net.eval()
        self.test_last_outputs = []
        with torch.no_grad():           
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    # subtract R from dist. This R value (for one-class method) was determined during training. 
                    # it is either directly the max "dist" score obtained from normal samples. or a value obtained from a regenerated  distribution 
                    # if the std_dev of dist scores was a small value (see the self.train() method)
                    # note that here R is subtracted NOT R**2 (as opposed to soft-boundary method) since the meaning of R is different here
                    scores = dist - self.R

                self.test_last_outputs.extend(outputs.data.tolist())
                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        self.num_defect =  labels.count(0)
        self.num_normal =  labels.count(1)
        labels = np.array(labels)
        scores = np.array(scores)       
        self.predictions = np.zeros_like(scores).astype(int)
        self.predictions[scores > 0] = 1
        self.test_AUC_CONT = roc_auc_score(labels, scores)
        self.pred_test_AUC = roc_auc_score(labels, self.predictions)
        logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_AUC_CONT))
        logger.info('Test set AUC (from predictions): {:.2f}%'.format(100. * self.pred_test_AUC))
        logger.info('Finished testing.')

    def initialize_pixel_wise_img_stats(self, train_loader: DataLoader):
        logger = logging.getLogger()
        if self.train_pix_mean is None or self.train_pix_std_dev is None:
            logger.info('Initializing pixelwise normal (non-defect) sample statistics.')
            self.init_train_img_stats(train_loader)

    def evaluate_sample(self, input_sample, net: BaseNet):
        logger = logging.getLogger()
        #self.initialize_pixel_wise_img_stats()
        # Set device for network
        net = net.to(self.device)
        # Testing
        logger.info('Starting evaluating the input sample...')
        start_time = time.time()
        idx_label_score = []
        net.eval()
        outputs = net(input_sample)
        dist = torch.sum((outputs - self.c) ** 2, dim=1)
        if self.objective == 'soft-boundary':
            scores = dist - self.R ** 2
        else:
            scores = dist - self.R

        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)
        logger.info('Finished testing.')
        return scores, self.test_time


    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)       
        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                #print('outputs : ', outputs)
                n_samples += outputs.shape[0]
                #print('outputs shape[0] : ', outputs.shape[0])
                c += torch.sum(outputs, dim=0)
                
        c /= n_samples
        #print('c:', c)

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def init_train_img_stats(self, train_loader: DataLoader):
            """Create and cache the pixelwise mean and std_dev of the training set (normal) images. 
            These values will be used to generate an image heat_map for defected images """
            n = 0
            tensor_shape = train_loader.dataset[0][0].shape
            S = torch.zeros(tensor_shape, dtype = torch.float32)
            m = torch.zeros(tensor_shape, dtype = torch.float32)
            with torch.no_grad():
                for data in train_loader:
                    # get the inputs of the batch
                    inputs, _, _ = data
                    inputs = inputs.to(self.device)               
                    # calculate pixel wise running mean and std_dev for input images
                    for input in inputs:
                        input = global_contrast_normalization(input, scale='l1')
                        n = n + 1
                        m_prev = m
                        m = m + (input - m) / n
                        S = S + (input - m) * (input - m_prev)

            self.train_pix_std_dev = torch.sqrt(S/n)
            self.train_pix_mean = m

            #return {'mean': self.train_pix_mean, 'std_dev': self.train_pix_std_dev}


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
