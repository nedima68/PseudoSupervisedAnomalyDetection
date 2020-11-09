import os
import cv2
from sklearn.cluster import KMeans
import numpy as np
from skimage.color import (separate_stains, combine_stains,  hdx_from_rgb, rgb_from_hdx, rgb_from_hed)
from skimage import data, segmentation, color
from skimage.future import graph
import torch
from torchvision import transforms, datasets
from ImgProcessing.NCutFeatureExtractor import generate_NCUT_segmented_image
from ImgProcessing.FeatureContourGenerator import draw_feature_contours
from utils.visualization.plot_images_grid import greatest_factor_pair
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

# define a simple function for displaying results 
def display_imgs(rows: int, cols: int, images: list, original_imgs: list):
    assert len(images) == len(original_imgs), "ERROR: length of the two lists must be equal"
    assert rows*cols >= 2*len(images), "ERROR: rows*cols must be more than  the sum of number of elements in the two lists"
    elem_num = len(images)
    figure(num=None, figsize=(18, 14), dpi=70, facecolor='w', edgecolor='k')
    img_idx = 0
    for i in range(0, rows*cols, 2):
        plt.subplot(rows, cols, i+1)        
        plt.imshow(cv2.cvtColor(original_imgs[img_idx], cv2.COLOR_BGR2RGB))
        plt.xticks([]), plt.yticks([])
        plt.subplot(rows, cols, i+2)
        plt.imshow(cv2.cvtColor(images[img_idx], cv2.COLOR_BGR2RGB))
        plt.xticks([]), plt.yticks([])
        img_idx += 1
        if img_idx >= elem_num:
            break
        
    plt.show()


def tensor_to_img(img, mean=0, std=1):
    img = np.transpose(img.detach().cpu().numpy(), (1, 2, 0))
    img = (img*std+ mean)*255
    img = img.astype(np.uint8)
    img = cv.cvtColor(img , cv2.COLOR_BGR2RGB)
    return img

def generate_NCUT(img):
    # first obtain superpixeled-image
    # labels_slic = segmentation.slic(img, compactness=13, n_segments=17) #good performance
    # labels_slic = segmentation.slic(img, compactness=12.4, n_segments=17.5) #also with seg = 17.4 better performance
    #labels_slic = segmentation.slic(img, compactness=20, n_segments=18.8) # good for fabric_03 
    labels_slic = segmentation.slic(img, compactness=18.4, n_segments=8.9) 
    #labels_slic = segmentation.slic(img, slic_zero=True)
    # for printing superpixeled image
    superpixels_slic = color.label2rgb(labels_slic, img, kind='avg')
    #superpixels_slic = color.label2rgb(labels_slic, img)
    
    # build rag (graph) out of the superpixeled image labels
    g = graph.rag_mean_color(img, labels_slic, connectivity=5, mode='similarity')
    #g = graph.rag_mean_color(img, superpixels_slic, connectivity=3, mode='similarity')
    #g = graph.rag_mean_color(img, labels_slic, mode='distance')
    # apply ncut algorithm to graph
    ncuts_labels = graph.cut_normalized(labels_slic, g)
    # replace each pixel region with its average color ('centroid')
    #ncuts_result = color.label2rgb(ncuts_labels, img, bg_label=0, colors=[(1, 1, 0), (0, 0.7, 1), (0.3, 0.7, 1)],kind='avg')
    ncuts_result = color.label2rgb(ncuts_labels, img, bg_label=0, kind='avg')
    return ncuts_result
    

if __name__ == '__main__':
    
    dataset_dir = 'E:/Anomaly Detection/test_patches/segmentation'
    dataset_dir = 'E:/temp/images/patches/SMatrix'
    images = {}
    ncut_images = []
    #import image subset by listing all images in dataset directory
    for filename in os.listdir(dataset_dir):
        #images[filename] = cv2.imread(dataset_dir+os.sep+filename, cv2.IMREAD_GRAYSCALE)
        images[filename] = cv2.imread(dataset_dir+os.sep+filename, cv2.IMREAD_COLOR)
        
    # obtain dict filename-image matrix
    print("Number of images: ", len(images))
    print("Shape of images: ", list(images.values())[0].shape)
    rows, cols = greatest_factor_pair(2 * len(images))
    for k, img in images.items():    
        ncuts_result_img = generate_NCUT(img)    
        # save result
        ncut_images.append(ncuts_result_img)

    display_imgs(rows, cols, ncut_images, list(images.values()))
    
   