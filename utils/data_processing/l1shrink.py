import numpy as np
import torch
from base.base_data_types import PatchSize

def shrink(epsilon, x, imsize = PatchSize(32,32), channels = 3):
    """
    @Original Author: Prof. Randy
    @Modified by: Chong Zhou
    update to python3: 03/15/2019
    modified by: M.Nedim Alpdemir on 05/06/2020
        modifications:  1 - make it work with pytorch tensors  
                        2 - our dataset provides a batch of 3 channel color images so reshape x to prepare for processing
    Args:
        epsilon: the shrinkage parameter (either a scalar or a vector)
        x: the vector to shrink on
        imsize: size of the image
        channels: number of color channels

    Returns:
        The shrunk vector
    """
    #output = np.array(x*0.)
    input = x.reshape(-1, imsize.x * imsize.y * channels)
    output = torch.zeros(x.shape)
    for i in range(input.shape[0]):
        for idx, ele in enumerate(input.shape[1]):
            if ele > epsilon:
                output[i, idx] = ele - epsilon
            elif ele < -epsilon:
                output[i, idx] = ele + epsilon
            else:
                output[i, idx] = 0.

    return output.reshape(-1, imsize.x, imsize.y, channels)