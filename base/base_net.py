import logging
import torch.nn as nn
import numpy as np

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)  or isinstance(m, nn.BatchNorm2d) or \
        isinstance(m, nn.ConvTranspose2d)  or isinstance(m, nn.BatchNorm1d):
        m.reset_parameters()

class BaseNet(nn.Module):
    """Base class for all neural networks."""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rep_dim = None  # representation dimensionality, i.e. dim of the last layer

    def forward(self, *input):
        """
        Forward pass logic
        :return: Network output
        """
        raise NotImplementedError

    def reset_params(self):
        self.apply(weight_reset)

    def summary(self):
        """Network summary."""
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])
        print('Trainable parameters: {}'.format(params))
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)
