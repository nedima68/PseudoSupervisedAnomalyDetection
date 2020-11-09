
import os
from sklearn.cluster import KMeans
import numpy as np
from skimage.color import (separate_stains, combine_stains,  hdx_from_rgb, rgb_from_hdx, rgb_from_hed)
from skimage import data, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from PIL import Image


def generate_NCUT_segmented_image(img):
    '''
    Generates a segmentation based on Normalized Cut algrithm
    Args:
        img: this is a 3 channel (RGB) openCV image

    Returns:
        segmented image 

    '''
       # assume a PILImage object
    # first obtain superpixeled-image
    # labels_slic = segmentation.slic(img, compactness=13, n_segments=17) #good performance
    # labels_slic = segmentation.slic(img, compactness=12.4, n_segments=17.5) #also with seg = 17.4 better performance
    #labels_slic = segmentation.slic(img, compactness=20, n_segments=18.8) # good for fabric_03 
    labels_slic = segmentation.slic(img, compactness=13.4, n_segments=11.5) 
    #labels_slic = segmentation.slic(img, slic_zero=True)
    # for printing superpixeled image
    superpixels_slic = color.label2rgb(labels_slic, img, kind='avg')
    
    # build rag (graph) out of the superpixeled image labels
    g = graph.rag_mean_color(img, labels_slic, connectivity=3, mode='similarity')
    #g = graph.rag_mean_color(img, labels_slic, mode='distance')
    # apply ncut algorithm to graph
    ncuts_labels = graph.cut_normalized(labels_slic, g)
    # replace each pixel region with its average color ('centroid')
    ncuts_result = color.label2rgb(ncuts_labels, img, bg_label=0, colors=('red','blue','green'),kind='avg')
    
    return ncuts_result
