import numpy as np
import torch
from base.base_data_types import PatchSize

def l21shrink(epsilon, x, imsize = PatchSize(32,32), channels = 3, purpose='ANOMALY_DETECTION'):
    """
    auther : Chong Zhou
    date : 10/20/2016
    update to python3: 03/15/2019
    modified by: M.Nedim Alpdemir on 05/06/2020
        modifications:  1 - make it work with pytorch tensors  
                        2 - our dataset provides a batch of 3 channel color images so reshape x to prepare for processing
                        3 - made it more intuitive and clear in terms of distinction between L2,1 regularization for anomaly detection
                        and L2,1 regularization for detecting bad features repeating in all of the instances (such as bad pixel due to camera failure etc.)
    Args:
        epsilon: the shrinkage parameter
        x: matrix to shrink on
        imsize: size of the image
        channels: number of color channels
    Ref:
        wiki Regularization: {https://en.wikipedia.org/wiki/Regularization_(mathematics)}
    Returns:
            The shrunk matrix
    """
    assert purpose in ['ANOMALY_DETECTION','BAD_FEATURE_DETECION'], 'ERROR: unrecognized purpose type given... Must be one of [ANOMALY_DETECTION, BAD_FEATURE_DETECION]'
    output = x.clone().detach().cpu().numpy()
    #eps = epsilon.detach().cpu().numpy()

    #output = x.clone()
    output = output.reshape(-1,  channels * imsize.x * imsize.y)
    if purpose == 'ANOMALY_DETECTION':
        output = output.T # take the transpose of the collection. 
    #norm = np.linalg.norm(x, ord=2, axis=0)
    norm = np.linalg.norm(output, ord = 2, axis = 0)
    #norm = output.norm(2, dim = 0)
    print(output.shape)
    for i in range(output.shape[1]): 
        if norm[i] > epsilon:
            for j in range(output.shape[0]): 
                output[j,i] = output[j,i] - epsilon * output[j,i] / norm[i]
        else:
            output[:,i] = 0.
        #print(i)

    if purpose == 'ANOMALY_DETECTION':
        output = output.T # take the transpose again.
    output = output.reshape(-1, channels, imsize.x, imsize.y)
    #return output.detach().cpu().numpy() # NOTE : we assume channel first image configuration
    return output