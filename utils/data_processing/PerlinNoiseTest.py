from utils.data_processing.perlin import PerlinNoiseFactory
from skimage.util import random_noise
import cv2 as cv
from matplotlib import pyplot as plt
#import pylab as plt  
from PIL import Image
import logging
import os
import numpy as np
from base.base_data_types import PatchSize
from utils.visualization.plot_images_grid import greatest_factor_pair

def julia_set_fractal_generator():
    # Python code for Julia Fractal 
    # setting the width, height and zoom  
    # of the image to be created 
    w, h, zoom = 32,32,1.2
   
    # creating the new image in RGB mode 
    bitmap = Image.new("L", (w, h), "white") 
  
    # Allocating the storage for the image and 
    # loading the pixel data. 
    pix = bitmap.load() 
     
    # setting up the variables according to  
    # the equation to  create the fractal 
    #cX = np.random.uniform(-0.69, -0.76)
    #cY = np.random.uniform(0.26, 0.29)
    zx_coeff = np.random.uniform(3.9, 5.8)
    zy_coeff = np.random.uniform(0.1, 0.8)
    sgn = np.random.choice([1,-1])

    cX, cY = -0.66, sgn*0.24015 # making cY negative or positive makes the defect right skewed or left skewed
    moveX, moveY = 0.0, 0.0
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

    np_img = np.asarray(bitmap)

    for x in range(w): 
        for y in range(h): 
            t = pix[x,y]
    
    return bitmap, zx_coeff, zy_coeff

def parametric_julia_set_fractal_generator(zx_range, zy_range):
    # Python code for Julia Fractal 
    # setting the width, height and zoom  
    # of the image to be created 
    w, h, zoom = 32,32,1.2
   
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


    cX, cY = -0.702, sgn*0.25715 # making cY negative or positive makes the defect right skewed or left skewed
    #moveX, moveY = 0.0, 0.0
    moveX, moveY = np.random.uniform(0.0, 0.7), np.random.uniform(0.0, 0.7)
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

def open_simplex():
    from opensimplex import OpenSimplex
    osimp = OpenSimplex()
    img = np.zeros((32,32))
    for x in range(32):
        for y in range(32):
           img[x,y] = int(255*osimp.noise2d(x, y))
     
    PIL_img = Image.fromarray(img)
    PIL_img.show()
    #print (tmp.noise2d(x=10, y=10))
    #new_img = cv.cvtColor(np.array(PIL_img), cv.COLOR_GRAY2BGR)
    #cv.imshow( 'sample image', new_img)
    #cv.waitKey( 0) # waits until a key is pressed.
    #cv.destroyAllWindows() # destroys the window showing image.

def generate_custom_noise(img, type = 'VERTICAL', img_size = PatchSize(32,32)):
    assert type in ['VERTICAL', 'HORIZONTAL', 'POINT'], "ERROR: Invalid noise type in julia set generator"
    np_img = np.asarray(img)
    max_p = np.amax(np_img)
    min_p = np.amin(np_img)
    gray_color_range = list(range(min_p, max_p))
    channel = np.random.uniform(10,18)
    thickness = int(channel*0.3)
    divergence = int(channel*0.5)
    length = np.random.uniform(7, img_size.x - 1)
    l_off_set = np.random.uniform(0, img_size.x - length)
    w_off_set = np.random.uniform(2, img_size.x - channel - 2)

    if type == 'VERTICAL':
        zx_range = [4.9, 5.3]
        zy_range = [0.2, 0.7]
    elif type == 'HORIZONTAL':
        x_0 = 0 + l_off_set
        x_n = x_0 + length
        y_0 = 0 + w_off_set + thickness
    elif type == 'POINT':
        zx_range = [3.7, 4.8]
        zy_range = [3.7, 4.8]

    
    for x in range(img_size.x):
        for y in range(img_size.y):
            if pix[x,y] < 150:
                im_pix = img.getpixel((x,y))
                img.putpixel((x, y), int((im_pix + 1.0*pix[x,y]) / 2.0))
                #img.putpixel((x, y), pix[x,y])
                #img.putpixel((x, y), int(np.abs(im_pix - min_p)))
                #img.putpixel((x, y), int((min_p + im_pix + 1*pix[x,y]) / 3.0))
                #img.putpixel((x, y), int((min_p + pix[x,y]) / 2.0))

    return img

def generate_julia_set_noise(img, type = 'VERTICAL', img_size = PatchSize(32,32)):
    assert type in ['VERTICAL', 'HORIZONTAL', 'POINT'], "ERROR: Invalid noise type in julia set generator"
    if type == 'VERTICAL':
        zx_range = [3.8, 4.9]
        zy_range = [0.3, 1.1]
    elif type == 'HORIZONTAL':
        zx_range = [0.3, 1.2]
        zy_range = [3.5, 4.2]
    elif type == 'POINT':
        zx_range = [3.7, 4.8]
        zy_range = [3.7, 4.8]

    noise_img = parametric_julia_set_fractal_generator(zx_range, zy_range)
    pix = noise_img.load()
    np_img = np.asarray(img)
    max_p = np.amax(np_img)
    min_p = np.amin(np_img)
    for x in range(img_size.x):
        for y in range(img_size.y):
            if pix[x,y] < 230:
                im_pix = img.getpixel((x,y))
                img.putpixel((x, y), int((im_pix + 1.0*pix[x,y]) / 2.0))
                #img.putpixel((x, y), pix[x,y])
                #img.putpixel((x, y), int(np.abs(im_pix - min_p)))
                #img.putpixel((x, y), int((min_p + im_pix + 1*pix[x,y]) / 3.0))
                #img.putpixel((x, y), int((min_p + pix[x,y]) / 2.0))

    return img

def generate_perlin_noise(img, x_range, y_range):
    pnf = PerlinNoiseFactory(2, octaves=4, tile=(4, 4))
    for x in x_range:
        for y in y_range:
            n = pnf(x/8, y/8)
            # generate a pixel value using perlin noise factory
            pix_val = int((n + 0.2 ) / 2 * 255 + 1.5 )
            # get the pixel of the original image at this location
            im_pix = img.getpixel((x,y))
            # blend the original pixel and created pixel using formula: new_pix = (alpha * old_pix + new_pix) / 2
            # alpha controls the weight of the original image
            img.putpixel((x, y), int((1.0*im_pix + pix_val) / 2))
    return img

def vertical_defect_julia_set(cv_img, im_size = PatchSize(32,32)):
    PIL_img = Image.fromarray(cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY))
    new_img = PIL_img.copy()
    new_img = generate_julia_set_noise(new_img, type='VERTICAL', img_size = im_size)
    new_img = cv.cvtColor(np.array(new_img), cv.COLOR_GRAY2BGR)
    return new_img

def horizontal_defect_julia_set(cv_img, im_size = PatchSize(32,32)):
    PIL_img = Image.fromarray(cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY))
    new_img = PIL_img.copy()
    new_img = generate_julia_set_noise(new_img, type='HORIZONTAL', img_size = im_size)
    new_img = cv.cvtColor(np.array(new_img), cv.COLOR_GRAY2BGR)
    return new_img

def point_defect_julia_set(cv_img, im_size = PatchSize(32,32)):
    PIL_img = Image.fromarray(cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY))
    new_img = PIL_img.copy()
    new_img = generate_julia_set_noise(new_img, type='POINT', img_size = im_size)
    new_img = cv.cvtColor(np.array(new_img), cv.COLOR_GRAY2BGR)
    return new_img


def vertical_defect(cv_img, im_size = PatchSize(32,32), thickness = 3):
    PIL_img = Image.fromarray(cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY))
    new_img = PIL_img.copy()
    x_1 = np.random.randint(0, im_size.x - (thickness + 1))
    x_2 = x_1 + thickness
    v_length = np.random.randint(thickness * 3, im_size.y-1)
    y_1 = np.random.randint(0, im_size.y - (v_length + 1))
    y_2 = y_1 + v_length
    x_range = range(x_1, x_2)
    y_range = range(y_1, y_2)
    new_img = generate_perlin_noise(new_img, x_range, y_range)
    new_img = cv.cvtColor(np.array(new_img), cv.COLOR_GRAY2BGR)
    return new_img

def horizontal_defect(cv_img, im_size = PatchSize(32,32), thickness = 3):
    PIL_img = Image.fromarray(cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY))
    new_img = PIL_img.copy()
    y_1 = np.random.randint(0, im_size.y - (thickness + 1))
    y_2 = y_1 + thickness
    h_length = np.random.randint(thickness * 3, im_size.x - 1)
    x_1 = np.random.randint(0, im_size.x - (h_length + 1))
    x_2 = x_1 + h_length
    x_range = range(x_1, x_2)
    y_range = range(y_1, y_2)
    new_img = generate_perlin_noise(new_img, x_range, y_range)
    new_img = cv.cvtColor(np.array(new_img), cv.COLOR_GRAY2BGR)
    return new_img

def spot_defect(cv_img, im_size = PatchSize(32,32), thickness = 4):
    PIL_img = Image.fromarray(cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY))
    new_img = PIL_img.copy()
    x_1 = np.random.randint(0, im_size.x - (thickness + 1))
    x_2 = x_1 + thickness
    y_1 = np.random.randint(0, im_size.y - (thickness + 1))
    y_2 = y_1 + thickness    
    x_range = range(x_1, x_2)
    y_range = range(y_1, y_2)
    new_img = generate_perlin_noise(new_img, x_range, y_range)
    new_img = cv.cvtColor(np.array(new_img), cv.COLOR_GRAY2BGR)
    return new_img

def poisson_noise_defect(cv_img, im_size = PatchSize(32,32)):
    new_img = cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY)
    poisson_noise = random_noise(np.array(new_img), mode='poisson')
    poisson_noise = np.clip(poisson_noise, 0., 1.)
    new_img = cv.normalize(src=poisson_noise, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    new_img = cv.cvtColor(np.asarray(new_img), cv.COLOR_GRAY2BGR)
    return new_img

def gaussian_noise_defect(cv_img, im_size = PatchSize(32,32)):
    new_img = cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY)
    gaussian_noise = random_noise(np.array(new_img), mode='gaussian')
    gaussian_noise = np.clip(gaussian_noise, 0., 1.)
    new_img = cv.normalize(src=gaussian_noise, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    new_img = cv.cvtColor(np.asarray(new_img), cv.COLOR_GRAY2BGR)
    return new_img

def localvar_noise_defect(cv_img, im_size = PatchSize(32,32)):
    new_img = cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY)
    localvar_noise = random_noise(np.array(new_img), mode='localvar')
    localvar_noise = np.clip(localvar_noise, 0., 1.)
    new_img = cv.normalize(src=localvar_noise, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    new_img = cv.cvtColor(np.asarray(new_img), cv.COLOR_GRAY2BGR)
    return new_img

def speckle_noise_defect(cv_img, im_size = PatchSize(32,32)):
    new_img = cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY)
    speckle_noise = random_noise(np.array(new_img), mode='speckle')
    new_img = cv.normalize(src=speckle_noise, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    new_img = cv.cvtColor(np.asarray(new_img), cv.COLOR_GRAY2BGR)
    return new_img

def test_skimage_noise():
    PLOT = True

    logging.basicConfig(level=logging.INFO)

    def_data_source_dir = 'E:/Anomaly Detection/NNetDatasets_reduced/AITEX/fabric_02/32x32/test/defect'
    nond_data_source_dir = 'E:/Anomaly Detection/NNetDatasets_reduced/AITEX/fabric_02/32x32/test/non_defect'
    big_defect_img_f = '0070_023_02__[7,114].png'
    line_defect_img_f = '0081_006_04__[4,65].png'
    # small_defect_img_f = '0046_019_04__[4,74].png' # fabric_04
    small_defect_img_f = '0027_019_02__[3,58].png' # fabric_02
    normal_img_f = '0005_000_04_N_[4,16].png'

    fab4_f1 = '0081_006_04__[0,15].png'
    fab4_f2 = '0046_019_04__[1,18].png'
    test_img = 'E:/Anomaly Detection/FabricDefectDetection/images/temp/test/non_defect/ben-esim.jpg'

    files = os.listdir(nond_data_source_dir)
    num = 0
    img_array = []
    for f in files:
        img = cv.imread(nond_data_source_dir + '/' + f)
        img_array.append(img)
        num += 1
        if num > 20:
            break
    img_array = np.array(img_array)
    #normal_img = np.mean(img_array, axis=0)
    normal_im_1 = img_array[0]
    normal_im_1 = normal_im_1.reshape(3072)
    normal_im_1 = normal_im_1.reshape(32,32,3)
    normal_img = cv.cvtColor(normal_im_1, cv.COLOR_BGR2RGB) / 255.0
    PIL_img = Image.fromarray(cv.cvtColor(normal_im_1, cv.COLOR_BGR2GRAY))
    #normal_img = cv.imread(nond_data_source_dir + '/' + normal_img)
    #defect_img = cv.imread(def_data_source_dir + '/' + small_defect_img_f, cv.IMREAD_GRAYSCALE)
    defect_img = cv.imread(def_data_source_dir + '/' + small_defect_img_f)
    gausian_noise = random_noise(normal_img, mode='gaussian')
    #gpil = Image.fromarray(cv.cvtColor(gausian_noise, cv.COLOR_BGR2RGB))
    localvar_noise = random_noise(normal_img, mode='localvar')
    poisson_noise = random_noise(normal_img, mode='poisson')
    salt_pepper_noise = random_noise(normal_img, mode='s&p')
    speckle_noise = random_noise(normal_img, mode='speckle', mean=0.2, var=0.1)
    row = 2
    col = 3
    if PLOT:
        imno = 1
        fig = plt.figure()                  
        plt.subplot(row, col,  imno)
        plt.imshow(normal_img, interpolation='nearest')
        plt.title("original")
               
        plt.subplot(row, col,  imno + 1)
        plt.imshow(gausian_noise, interpolation='nearest')
        plt.title("gaussian")

        plt.subplot(row, col,  imno + 2)
        plt.imshow(localvar_noise, interpolation='nearest')
        plt.title("localvar")

        plt.subplot(row, col,  imno + 3)
        plt.imshow(poisson_noise)
        plt.title("poisson")

        plt.subplot(row, col, imno + 4)
        plt.imshow(salt_pepper_noise)
        plt.title("salt_pepper")

        plt.subplot(row, col, imno + 5)
        plt.imshow(speckle_noise)
        plt.title("speckle")

        plt.show()


def test_defect_as_perlin_noise():
    PLOT = True
    logging.basicConfig(level=logging.INFO)

    def_data_source_dir = 'E:/Anomaly Detection/NNetDatasets_reduced/AITEX/fabric_02/32x32/test/defect'
    nond_data_source_dir = 'E:/Anomaly Detection/NNetDatasets_reduced/AITEX/fabric_02/32x32/test/non_defect'
    files = os.listdir(nond_data_source_dir)
    num = 0
    img_array = []
    for f in files:
        img = cv.imread(nond_data_source_dir + '/' + f)
        img_array.append(img)
        num += 1
        if num > 20:
            break
    img_array = np.array(img_array)
    normal_im_1 = img_array[0]
    #normal_img = cv.cvtColor(normal_im_1, cv.COLOR_BGR2RGB)
    #PIL_img = Image.fromarray(cv.cvtColor(normal_im_1, cv.COLOR_BGR2GRAY))  
    
    img_set = {'vertical':[], 'horizontal':[], 'spot':[], 'patch':[], 'poisson':[], 'gaussian':[], 'speckle':[], 'localvar':[]}
    for i in range(4):
        new_img = horizontal_defect(normal_im_1, thickness = 5)        
        img_set['horizontal'].append(new_img)
    for i in range(4):
        new_img = vertical_defect(normal_im_1, thickness = 5)        
        img_set['vertical'].append(new_img)
    for i in range(4):
        new_img = spot_defect(normal_im_1,thickness = 8)        
        img_set['spot'].append(new_img)

    for i in range(4):
        new_img = spot_defect(normal_im_1,thickness = 30)        
        img_set['patch'].append(new_img)

    for i in range(2):
        new_img = poisson_noise_defect(normal_im_1)
        img_set['poisson'].append(new_img)

    img_set['gaussian'].append(gaussian_noise_defect(normal_im_1))
    img_set['speckle'].append(speckle_noise_defect(normal_im_1))
    img_set['localvar'].append(localvar_noise_defect(normal_im_1))

    total_img_num = 0

    for v in img_set.values():
        total_img_num += len(v)
    
    row, col = greatest_factor_pair(total_img_num)
    if PLOT:
        imno = 1
        fig = plt.figure(figsize=(6, 6), dpi =100)
        for def_type, img_list in img_set.items():            
            for i, img in enumerate(img_list):
                plt.subplot(row, col, imno, xticks=[], yticks= [])
                plt.imshow(img)
                plt.title(def_type + '-'+ str(i), fontsize = 11)     
                imno += 1
        plt.tight_layout(pad = 0.2)
        plt.show()

def test_defect_as_julia_set_noise():
    PLOT = True
    logging.basicConfig(level=logging.INFO)

    def_data_source_dir = 'E:/Anomaly Detection/NNetDatasets_reduced/AITEX/fabric_01/32x32/test/defect'
    nond_data_source_dir = 'E:/Anomaly Detection/NNetDatasets_reduced/AITEX/fabric_01/32x32/test/non_defect'
    files = os.listdir(nond_data_source_dir)
    num = 0
    img_array = []
    for f in files:
        img = cv.imread(nond_data_source_dir + '/' + f)
        img_array.append(img)
        num += 1
        if num > 20:
            break
    img_array = np.array(img_array)
    normal_im_1 = img_array[0]
    #normal_img = cv.cvtColor(normal_im_1, cv.COLOR_BGR2RGB)
    #PIL_img = Image.fromarray(cv.cvtColor(normal_im_1, cv.COLOR_BGR2GRAY))  
    
    img_set = {'vertical':[], 'horizontal':[], 'spot':[], 'patch':[], 'poisson':[], 'gaussian':[], 'speckle':[], 'localvar':[]}
    for i in range(4):
        new_img = horizontal_defect_julia_set(normal_im_1)        
        img_set['horizontal'].append(new_img)
    for i in range(4):
        new_img = vertical_defect_julia_set(normal_im_1)        
        img_set['vertical'].append(new_img)
    for i in range(4):
        new_img = point_defect_julia_set(normal_im_1)        
        img_set['spot'].append(new_img)

    for i in range(4):
        new_img = spot_defect(normal_im_1,thickness = 30)        
        img_set['patch'].append(new_img)

    total_img_num = 0

    for v in img_set.values():
        total_img_num += len(v)
    
    row, col = greatest_factor_pair(total_img_num)
    if PLOT:
        imno = 1
        fig = plt.figure(figsize=(6, 6), dpi =100)
        for def_type, img_list in img_set.items():            
            for i, img in enumerate(img_list):
                plt.subplot(row, col, imno, xticks=[], yticks= [])
                plt.imshow(img)
                plt.title(def_type + '-'+ str(i), fontsize = 11)     
                imno += 1
        plt.tight_layout(pad = 0.2)
        plt.show()


def test_perlin_noise():
    size = 200
    res = 40
    frames = 10
    frameres = 5
    space_range = size//res
    frame_range = frames//frameres

    pnf = PerlinNoiseFactory(3, octaves=2, tile=(space_range, space_range, frame_range))

    for t in range(frames):
        img = Image.new('L', (size, size))
        for x in range(size):
            for y in range(size):
                n = pnf(x/res, y/res, t/frameres)
                #img.putpixel((x, y), int((n + 1) / 2 * 255 + 0.5))
                img.putpixel((x, y), int(n/2*255))

        img.save("noiseframe{:03d}.png".format(t))
        print(t)

def julia_set_test():
    for  t in range(15):
        img, zx,zy = julia_set_fractal_generator()
        img.save("E:/temp/noiseframe{:03d}_{:.2f}_{:.2f}.png".format(t, zx, zy))

if __name__ == '__main__':
    #test_perlin_noise()
    #test_defect_as_perlin_noise()
    test_defect_as_julia_set_noise()
    #test_skimage_noise()
    #julia_set_test()
    #open_simplex()
    