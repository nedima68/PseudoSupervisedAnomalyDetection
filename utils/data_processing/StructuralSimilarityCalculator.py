from skimage.measure import compare_ssim, compare_psnr
from skimage import data, img_as_float
#from skimage.metrics import structural_similarity 
import logging
import time
import numpy as np



def calc_ssim(x, x_rec):
    """Calculates SSIM between x and y.
    x should be a batch of original images. y should be a batch of reconstructed images.

    :param x: 4D tensor in form of batch_size x img_width x img_height x img_channels.
    :param x_rec: 4D tensor in form of batch_size x img_width x img_height x img_channels.
    :return: Numpy array with ssim score for each image in the given batch.
    """
    logger = logging.getLogger()
    res = []
    num_img = x_rec.shape[0]
    #logger.info('Calculating SSIM using autoencoder ...')
    for i in range(num_img):
        temp_x = x[i, ...]
        temp_rec = x_rec[i, ...]
        temp_x = img_as_float(temp_x.cpu().detach().numpy())
        if (len(temp_x.shape) > 2):
            # then this is an image with channels. e.g. 3x32x32. for 32x32 image with 3 color channels
            # we convert this to a channel last format i.e. 32x32x3
            # if this was a gray scale image e.g. 28x28 as in MNIST then no need to reshape
            temp_x = np.reshape(temp_x,(temp_x.shape[2],temp_x.shape[1], temp_x.shape[0]))

        temp_rec = img_as_float(temp_rec.cpu().detach().numpy())
        if (len(temp_rec.shape) > 2):
            # then this is an image with channels. e.g. 3x32x32. for 32x32 image with 3 color channels
            # we convert this to a channel last format i.e. 32x32x3
            temp_rec = np.reshape(temp_rec,(temp_rec.shape[2],temp_rec.shape[1], temp_rec.shape[0]))

        temp = compare_ssim(temp_x, temp_rec, multichannel=True, gaussian_weights=True)
        res.append(temp)

    return res
