import cv2
from PIL import Image
import numpy as np
from datasets.preprocessing import global_contrast_normalization

def create_heat_map_for_img(input_img, ref_img_stats, show = False):
    # assert that the ref_img_stats contains two image size tensors 
    # one containing pixel wise mean values the other containing pixel wise std_dev values 
    assert(input_img.shape == ref_img_stats[0].shape)
    assert(input_img.shape == ref_img_stats[1].shape)
    mean_values = ref_img_stats[0]
    std_dev_values = ref_img_stats[1]
    input_img = global_contrast_normalization(input_img, scale='l1')
    diff = np.abs(input_img - mean_values)
    # assuming that the image has the shape (3,32,32) e.g. 3 channel color image of size 32x32
    w = input_img.shape[1]
    h = input_img.shape[2]
    channels = input_img.shape[0]
    DARK_BLUE = [0,0,139]
    RED = [255,0,0]
    image = np.zeros((w,h,channels), dtype="uint8")
    image[np.where((image==[0,0,0]).all(axis=2))] = DARK_BLUE
    # change axis from 3,32,32 to 32,32,3, i.e. channels last
    diff = np.einsum('ijk->jki',diff)
    std_dev_values = np.einsum('ijk->jki',std_dev_values)
    for y in range(h):
        for x in range(w):
            delta = np.mean(diff[x][y])
            if delta > 2.2*np.mean(std_dev_values[x][y]): 
                image[x][y][0] = RED[0]
                image[x][y][1] = 0
                image[x][y][2] = 0
                #image[x][y][1] = 255 - int(30 * delta)
                #image[x][y][2] = 255 - int(30 * delta)
    if show:
        imno = 1
        fig = plt.figure()

        plt.subplot(2, 3, imno)
        plt.imshow(input_img)
        plt.title("Input image")

        plt.subplot(2, 3, imno+1)
        plt.imshow(image)
        plt.title("defect heat map")
        plt.show()

    return image


