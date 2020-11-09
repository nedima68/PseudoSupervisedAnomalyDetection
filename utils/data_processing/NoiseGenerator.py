from utils.data_processing.perlin import PerlinNoiseFactory
from skimage.util import random_noise
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image
import logging
import os
import numpy as np
from base.base_data_types import PatchSize
from utils.visualization.plot_images_grid import greatest_factor_pair

def generate_perlin_noise(img, x_range, y_range):
    pnf = PerlinNoiseFactory(2, octaves=4, tile=(4, 4))
    for x in x_range:
        for y in y_range:
            n = pnf(x/7, y/7)
            # generate a pixel value using perlin noise factory
            pix_val = int((n + 0.2 ) / 2 * 255 + 1.5 )
            # get the pixel of the original image at this location
            im_pix = img.getpixel((x,y))
            # blend the original pixel and created pixel using formula: new_pix = (alpha * old_pix + new_pix) / 2
            # alpha controls the weight of the original image
            img.putpixel((x, y), int((1.1*im_pix + pix_val) / 2))
    return img


def parametric_julia_set_fractal_generator(zx_range, zy_range):
    # Python code for Julia Fractal 
    # setting the width, height and zoom  
    # of the image to be created 
    w, h, zoom = 32,32,1.0
   
    # creating the new image in RGB mode 
    bitmap = Image.new("L", (w, h), "white") 
  
    # Allocating the storage for the image and 
    # loading the pixel data. 
    pix = bitmap.load() 
     
    # setting up the variables according to  
    # the equation to  create the fractal 
    #cX = np.random.uniform(-0.69, -0.76)
    #cY = np.random.uniform(0.26, 0.29)
    zx_coeff = np.random.uniform(zx_range[0], zx_range[1])
    zy_coeff = np.random.uniform(zy_range[0], zy_range[1])
    sgn = np.random.choice([1,-1])

    cX, cY = -0.691, sgn*0.27015 # making cY negative or positive makes the defect right skewed or left skewed
    #cX, cY = 0.0, -0.8 # making cY negative or positive makes the defect right skewed or left skewed
    #moveX, moveY = 0.0, 0.0
    moveX, moveY = np.random.uniform(-0.6, 0.9), np.random.uniform(-0.6, 0.9)
    maxIter = 255
   
    for x in range(w): 
        for y in range(h): 
            zx = zx_coeff*(x - w/2)/(0.5*zoom*w) + moveX 
            zy = zy_coeff*(y - h/2)/(0.5*zoom*h) + moveY 
            i = maxIter 
            while zx*zx + zy*zy < 4 and i > 1: 
                tmp = zx*zx - zy*zy + cX 
                zy,zx = 2.0*zx*zy + cY, tmp 
                i -= 1
  
            # convert byte to RGB (3 bytes), kinda  
            # magic to get nice colors 
            #pix[x,y] = (i << 21) + (i << 10) + i*8
            pix[x,y] = i
  
    # to display the created fractal 
    #bitmap.show() 
    
    return bitmap

def generate_julia_set_noise(img, type = 'VERTICAL', img_size = PatchSize(32,32)):
    assert type in ['VERTICAL', 'HORIZONTAL', 'POINT'], "ERROR: Invalid noise type in julia set generator"
    if type == 'VERTICAL':
        zx_range = [4.9, 5.3]
        zy_range = [0.2, 0.7]
    elif type == 'HORIZONTAL':
        zx_range = [0.3, 1.2]
        zy_range = [3.5, 4.2]
    elif type == 'POINT':
        zx_range = [3.2, 3.8]
        zy_range = [3.2, 3.8]

    noise_img = parametric_julia_set_fractal_generator(zx_range, zy_range)
    pix = noise_img.load()
    np_img = np.asarray(img)
    max_p = np.amax(np_img)
    min_p = np.amin(np_img)
    for x in range(img_size.x):
        for y in range(img_size.y):
            if pix[x,y] < 200:
                im_pix = img.getpixel((x,y))
                #img.putpixel((x, y), int((im_pix + 1.5*pix[x,y]) / 2))
                #img.putpixel((x, y), pix[x,y])
                #img.putpixel((x, y), int(np.abs(im_pix - min_p)))
                img.putpixel((x, y), int((1.1*min_p + im_pix + pix[x,y]) / 3.0))

    return img



def vertical_defect_julia_set(cv_img, im_size = PatchSize(32,32)):
    if len(cv_img.shape) > 2:
        PIL_img = Image.fromarray(cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY))
    else:
        PIL_img = Image.fromarray(cv_img)
    new_img = PIL_img.copy()
    new_img = generate_julia_set_noise(new_img, type='VERTICAL', img_size = im_size)

    if len(cv_img.shape) > 2:
        new_img = cv.cvtColor(np.array(new_img), cv.COLOR_GRAY2BGR)
    else:
        new_img = np.array(new_img)
    return new_img

def horizontal_defect_julia_set(cv_img, im_size = PatchSize(32,32)):
    if len(cv_img.shape) > 2:
        PIL_img = Image.fromarray(cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY))
    else:
        PIL_img = Image.fromarray(cv_img)
    new_img = PIL_img.copy()
    new_img = generate_julia_set_noise(new_img, type='HORIZONTAL', img_size = im_size)
    if len(cv_img.shape) > 2:
        new_img = cv.cvtColor(np.array(new_img), cv.COLOR_GRAY2BGR)
    else:
        new_img = np.array(new_img)
    return new_img

def point_defect_julia_set(cv_img, im_size = PatchSize(32,32)):
    if len(cv_img.shape) > 2:
        PIL_img = Image.fromarray(cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY))
    else:
        PIL_img = Image.fromarray(cv_img)
    new_img = PIL_img.copy()
    new_img = generate_julia_set_noise(new_img, type='POINT', img_size = im_size)
    if len(cv_img.shape) > 2:
        new_img = cv.cvtColor(np.array(new_img), cv.COLOR_GRAY2BGR)
    else:
        new_img = np.array(new_img)
    return new_img

def vertical_defect(cv_img, im_size = PatchSize(32,32), thickness = 3):
    if len(cv_img.shape) > 2:
        PIL_img = Image.fromarray(cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY))
    else:
        PIL_img = Image.fromarray(cv_img)
    new_img = PIL_img.copy()
    x_1 = np.random.randint(0, im_size.x - (thickness + 1))
    x_2 = x_1 + thickness
    v_length = np.random.randint(thickness * 3, im_size.y-1)
    y_1 = np.random.randint(0, im_size.y - (v_length + 1))
    y_2 = y_1 + v_length
    x_range = range(x_1, x_2)
    y_range = range(y_1, y_2)
    new_img = generate_perlin_noise(new_img, x_range, y_range)
    if len(cv_img.shape) > 2:
        new_img = cv.cvtColor(np.array(new_img), cv.COLOR_GRAY2BGR)
    else:
        new_img = np.array(new_img)
    return new_img

def horizontal_defect(cv_img, im_size = PatchSize(32,32), thickness = 3):
    if len(cv_img.shape) > 2:
        PIL_img = Image.fromarray(cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY))
    else:
        PIL_img = Image.fromarray(cv_img)
    new_img = PIL_img.copy()
    y_1 = np.random.randint(0, im_size.y - (thickness + 1))
    y_2 = y_1 + thickness
    h_length = np.random.randint(thickness * 3, im_size.x - 1)
    x_1 = np.random.randint(0, im_size.x - (h_length + 1))
    x_2 = x_1 + h_length
    x_range = range(x_1, x_2)
    y_range = range(y_1, y_2)
    new_img = generate_perlin_noise(new_img, x_range, y_range)
    if len(cv_img.shape) > 2:
        new_img = cv.cvtColor(np.array(new_img), cv.COLOR_GRAY2BGR)
    else:
        new_img = np.array(new_img)
    return new_img

def spot_defect(cv_img, im_size = PatchSize(32,32), thickness = 4):
    if len(cv_img.shape) > 2:
        PIL_img = Image.fromarray(cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY))
    else:
        PIL_img = Image.fromarray(cv_img)
    new_img = PIL_img.copy()
    x_1 = np.random.randint(0, im_size.x - (thickness + 1))
    x_2 = x_1 + thickness
    y_1 = np.random.randint(0, im_size.y - (thickness + 1))
    y_2 = y_1 + thickness    
    x_range = range(x_1, x_2)
    y_range = range(y_1, y_2)
    new_img = generate_perlin_noise(new_img, x_range, y_range)
    if len(cv_img.shape) > 2:
        new_img = cv.cvtColor(np.array(new_img), cv.COLOR_GRAY2BGR)
    else:
        new_img = np.array(new_img)
    return new_img

def poisson_noise_defect(cv_img):
    if len(cv_img.shape) > 2:
        new_img = cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY)
    else:
        new_img = cv_img
    poisson_noise = random_noise(np.array(new_img), mode='poisson')
    poisson_noise = np.clip(poisson_noise, 0., 1.)
    new_img = cv.normalize(src=poisson_noise, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    if len(cv_img.shape) > 2:
        new_img = cv.cvtColor(np.asarray(new_img), cv.COLOR_GRAY2BGR)
    else:
        new_img = np.asarray(new_img)
    return new_img

def gaussian_noise_defect(cv_img):
    if len(cv_img.shape) > 2:
        new_img = cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY)
    else:
        new_img = cv_img
    gaussian_noise = random_noise(np.array(new_img), mode='gaussian')
    gaussian_noise = np.clip(gaussian_noise, 0., 1.)
    new_img = cv.normalize(src=gaussian_noise, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    if len(cv_img.shape) > 2:
        new_img = cv.cvtColor(np.asarray(new_img), cv.COLOR_GRAY2BGR)
    else:
        new_img = np.asarray(new_img)
    return new_img

def localvar_noise_defect(cv_img):
    if len(cv_img.shape) > 2:
        new_img = cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY)
    else:
        new_img = cv_img
    localvar_noise = random_noise(np.array(new_img), mode='localvar')
    localvar_noise = np.clip(localvar_noise, 0., 1.)
    new_img = cv.normalize(src=localvar_noise, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    if len(cv_img.shape) > 2:
        new_img = cv.cvtColor(np.asarray(new_img), cv.COLOR_GRAY2BGR)
    else:
        new_img = np.asarray(new_img)
    return new_img

def speckle_noise_defect(cv_img):
    if len(cv_img.shape) > 2:
        new_img = cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY)
    else:
        new_img = cv_img
    speckle_noise = random_noise(np.array(new_img), mode='speckle')
    new_img = cv.normalize(src=speckle_noise, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    if len(cv_img.shape) > 2:
        new_img = cv.cvtColor(np.asarray(new_img), cv.COLOR_GRAY2BGR)
    else:
        new_img = np.asarray(new_img)
    return new_img

