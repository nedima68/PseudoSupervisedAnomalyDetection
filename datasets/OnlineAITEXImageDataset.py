
from torch.utils.data import Subset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from base.torchvision_dataset import TorchvisionDataset, CustomImageFolder
from datasets.preprocessing import get_target_label_idx, global_contrast_normalization, calculate_std_mean
from base.base_data_types import  PatchSize
from utils.visualization.display_sample_patches import tensor_to_img
from utils.data_processing.NoiseGenerator import horizontal_defect, vertical_defect, horizontal_defect_julia_set, vertical_defect_julia_set, point_defect_julia_set,spot_defect, gaussian_noise_defect
import os
import cv2 as cv
import numpy as np
import torch
import logging

def get_patches_distinct(source_root, max_normal_sample_num, xPixelSize, yPixelSize, img_color):
    """
    Original AITEX images comes with separate image files for defected fabrics and non-defected samples.
    This function extracts patches from original AITEX images using defect images for defect samples and non-defect images for normal (non-defect) samples. 
    This may sometimes be more preferable to make sure that non-defect samples are more uniform.
    """
    assert img_color in ['COLOR','GRAY_SCALE']
    data_source_dir = source_root + '/defect_samples'
    mask_source_dir = source_root + '/mask' 
   
    files = os.listdir(mask_source_dir)
    mask_images = []
    defect_images = []
    for f in files:
        mask_img = cv.imread(mask_source_dir + '/' + f, cv.IMREAD_GRAYSCALE)
        mask_images.append(mask_img)
        defect_f_name = f[:f.find('_mask')]  + f[f.find('.'):]
        if img_color == 'GRAY_SCALE':
            defect_img = cv.imread(data_source_dir + '/' + defect_f_name, cv.IMREAD_GRAYSCALE)
        else:
            defect_img = cv.imread(data_source_dir + '/' + defect_f_name)

        defect_images.append(defect_img)
    
    defect_samples = []
    non_defect_samples = []
    defect_index = 0
    non_defect_index = 0

    img_invalid_texture_offset = 1350
    for im_idx, mask_img in enumerate(mask_images):
        size = mask_img.shape
        img_height = size[0]
        img_width = size[1]
        real_img = defect_images[im_idx]
        if ((img_height % yPixelSize == 0) and (img_width % xPixelSize == 0)):
            i = 0
            j = 0
            for y in range(0, img_height, yPixelSize):
                i = 0
                for x  in range(0, img_width, xPixelSize):    
                    mask_patch = mask_img[y:y+yPixelSize, x:x+xPixelSize] # crop the mask image
                    img_patch = real_img[y:y+yPixelSize, x:x+xPixelSize] # crop the real image
                    # check whether there are any white pixels in this patch. If so this is a defect. this function returns an numpy array
                    white_pix_count = np.argwhere(mask_patch > 0).size 
                    if white_pix_count > 1:
                        label = 0
                        defect_samples.append((img_patch, label, (j,i)))
                        defect_index += 1
                        #cv.imwrite('E:/temp/samples/def/def_'+str(defect_index)+'.png', img_patch)
                    i += 1
                j += 1
        elif (img_height % yPixelSize != 0):
            print("Error: Please use another divisor for the column split.")
            return 
        elif (img_width % xPixelSize != 0):
            print("Error: Please use another divisor for the row split.")        

    
    non_def_data_source_dir = source_root + '/no_defect_samples'
    files = os.listdir(non_def_data_source_dir)
    non_defect_images = []
    for f in files:
        if img_color == 'GRAY_SCALE':
            non_defect_img = cv.imread(non_def_data_source_dir + '/' + f, cv.IMREAD_GRAYSCALE)
        else:
            non_defect_img = cv.imread(non_def_data_source_dir + '/' + f)

        non_defect_images.append(defect_img)

    im_idx = 0
    need_more_patches = True
    while (need_more_patches):
        img = non_defect_images[im_idx]
        size = img.shape
        img_height = size[0]
        img_width = size[1]
        i = 0
        j = 0
        for y in range(0, img_height, yPixelSize):
            i = 0
            for x  in range(0, img_width, xPixelSize):                           
                img_patch = img[y:y+yPixelSize, x:x+xPixelSize] # crop the real image                        
                label = 1
                if (non_defect_index < max_normal_sample_num) and (x > img_invalid_texture_offset):
                    non_defect_samples.append((img_patch, label, (j,i)))
                    non_defect_index += 1
                    #cv.imwrite('E:/temp/samples/non_def/ndef_'+str(non_defect_index)+'.png', img_patch)

                i += 1
            j += 1

        im_idx += 1
        if (non_defect_index >= max_normal_sample_num):
            need_more_patches = False

    return defect_samples, non_defect_samples



def get_patches_mixed(source_root, max_normal_sample_num, xPixelSize, yPixelSize, img_color):
    """
    This function extracts patches from original AITEX images first using defects images for both defect and normal samples, then if not sufficient 
    uses non-defect images to extract normal(non-defect) samples. Since defect parts (pixels) in the image can  unambiguosly be identified (using mask images)
    original images that contain defects can also be used to extract non-defect (normal) samples. This may sometimes be more desirable to simulate the rolling 
    behaviour of fabric inspection machines
    """
    assert img_color in ['COLOR','GRAY_SCALE']
    data_source_dir = source_root + '/defect_samples'
    mask_source_dir = source_root + '/mask' 
   
    files = os.listdir(mask_source_dir)
    mask_images = []
    defect_images = []
    for f in files:
        mask_img = cv.imread(mask_source_dir + '/' + f, cv.IMREAD_GRAYSCALE)
        mask_images.append(mask_img)
        defect_f_name = f[:f.find('_mask')]  + f[f.find('.'):]
        if img_color == 'GRAY_SCALE':
            defect_img = cv.imread(data_source_dir + '/' + defect_f_name, cv.IMREAD_GRAYSCALE)
        else:
            defect_img = cv.imread(data_source_dir + '/' + defect_f_name)

        defect_images.append(defect_img)
    
    defect_samples = []
    non_defect_samples = []
    defect_index = 0
    non_defect_index = 0

    img_invalid_texture_offset = 1350
    for im_idx, mask_img in enumerate(mask_images):
        size = mask_img.shape
        img_height = size[0]
        img_width = size[1]
        real_img = defect_images[im_idx]
        if ((img_height % yPixelSize == 0) and (img_width % xPixelSize == 0)):
            i = 0
            j = 0
            for y in range(0, img_height, yPixelSize):
                i = 0
                for x  in range(0, img_width, xPixelSize):    
                    mask_patch = mask_img[y:y+yPixelSize, x:x+xPixelSize] # crop the mask image
                    img_patch = real_img[y:y+yPixelSize, x:x+xPixelSize] # crop the real image
                    # check whether there are any white pixels in this patch. If so this is a defect. this function returns an numpy array
                    white_pix_count = np.argwhere(mask_patch > 0).size 
                    if white_pix_count > 1:
                        label = 0
                        defect_samples.append((img_patch, label, (j,i)))
                        defect_index += 1
                        #cv.imwrite('E:/temp/samples/def/def_'+str(defect_index)+'.png', img_patch)
                    else:
                        label = 1
                        if (non_defect_index < max_normal_sample_num) and (x > img_invalid_texture_offset):
                            non_defect_samples.append((img_patch, label, (j,i)))
                            non_defect_index += 1
                            #cv.imwrite('E:/temp/samples/non_def/ndef_'+str(non_defect_index)+'.png', img_patch)

                    i += 1
                j += 1
        elif (img_height % yPixelSize != 0):
            print("Error: Please use another divisor for the column split.")
            return 
        elif (img_width % xPixelSize != 0):
            print("Error: Please use another divisor for the row split.")        

    print ("extracted {} defect {} non_defect image patches".format(defect_index, non_defect_index))
    if non_defect_index < max_normal_sample_num:
        # coludn't collect sufficient non defect patches from the clean parts of the defect images
        # collect some more from the non-defect only images
        print (" need {} more non-defect patches. extracting ...".format(max_normal_sample_num - non_defect_index))
        non_def_data_source_dir = source_root + '/no_defect_samples'
        files = os.listdir(non_def_data_source_dir)
        non_defect_images = []
        for f in files:
            if img_color == 'GRAY_SCALE':
                non_defect_img = cv.imread(non_def_data_source_dir + '/' + f, cv.IMREAD_GRAYSCALE)
            else:
                non_defect_img = cv.imread(non_def_data_source_dir + '/' + f)

            non_defect_images.append(defect_img)

        im_idx = 0
        need_more_patches = True
        while (need_more_patches):
            img = non_defect_images[im_idx]
            size = img.shape
            img_height = size[0]
            img_width = size[1]
            i = 0
            j = 0
            for y in range(0, img_height, yPixelSize):
                i = 0
                for x  in range(0, img_width, xPixelSize):                           
                    img_patch = img[y:y+yPixelSize, x:x+xPixelSize] # crop the real image                        
                    label = 1
                    if (non_defect_index < max_normal_sample_num) and (x > img_invalid_texture_offset):
                        non_defect_samples.append((img_patch, label, (j,i)))
                        non_defect_index += 1
                        #cv.imwrite('E:/temp/samples/non_def/ndef_'+str(non_defect_index)+'.png', img_patch)

                    i += 1
                j += 1

            im_idx += 1
            if (non_defect_index >= max_normal_sample_num):
                need_more_patches = False

    return defect_samples, non_defect_samples

class OnlineImagePatchFeeder:
    """
        Torchvision class with patch of __getitem__ method to also return the index of a data sample.
     
        This class generates normal and anomalous (defected) image patches of a given size (e.g. 32x32) using original images contained in the AITEX dataset.
        It also generates synthetic defects by decorating normal patches obtained from original AITEX images. 

    """

    def __init__(self, root, transform, target_transform, mode = 'train', max_normal_sample_num = 1600, gen_synthetic_defect = False, synth_def_label = -1, defect_ratio = 5, noise_type = 'structured',
                 patch_size = PatchSize(32,32), collect_stats = False, img_color = 'COLOR', *args, **kwargs):
        assert mode in ['train','test'], "ERROR: invalid mode given ! must be one of 'train or 'test' "
        logger = logging.getLogger()
        self.mode = mode
        self.synth_def_label = synth_def_label
        self.patch_size = patch_size
        self.gen_synthetic_defects = gen_synthetic_defect
        self.defect_ratio = defect_ratio
        self.transform = transform
        self.target_transform = target_transform
        #defect_samples, non_defect_samples = get_patches_distinct(root, max_normal_sample_num, xPixelSize = patch_size.x, yPixelSize = patch_size.y, img_color = img_color)
        defect_samples, non_defect_samples = get_patches_mixed(root, max_normal_sample_num, xPixelSize = patch_size.x, yPixelSize = patch_size.y, img_color = img_color)
        logger.info("Total number of extracted samples; [defect, normal] = [{}, {}]".format(len(defect_samples), len(non_defect_samples)))
        self.samples = self.split_data(defect_samples, non_defect_samples)
        logger.info("After data splitting: Dataset Mode [{}], num usable samples: {}".format(self.mode, len(self.samples)))

        self.sample_mean = 0.0
        self.sample_stddev = 0.0
        if self.mode == 'train' and collect_stats:
            #tx = transforms.ToTensor()
            imgs, lbls, indices = zip(*self.samples)
            ar  = np.asarray(imgs)
            self.sample_mean = np.mean(ar, axis=(0,1,2))
            self.sample_stddev = np.std(ar, axis=(0,1,2))

        if self.mode == 'train' and self.gen_synthetic_defects:
            self.generate_synthetic_defects(noise_type = noise_type)

    def generate_synthetic_defects(self, noise_type):
        """
        This function generates synthetic defects. Two different alternatives can be used for this purpose.
        1 - Perlin Noise can be used for wide area defects 
            see https://en.wikipedia.org/wiki/Perlin_noise
            we used the algorithm implemented by https://gist.github.com/eevee/26f547457522755cb1fb8739d0ea89a1
        2 - JuliaSet Fractals can be used dor localized defects, such as vertical, horizontal or point defects
            we developed a parameterized version of the algorithm available  from https://www.geeksforgeeks.org/julia-fractal-python/
        """
        assert noise_type in ['statistical', 'structured'], "ERROR: wrong noise type is given"
        
        logger = logging.getLogger()
        # find the number of defects such that that number is the specified noise_ratio of the total_sample set after the defects are added
        synth_defect_num = int(len(self.samples) * (self.defect_ratio / 100) / (1 - self.defect_ratio / 100))
        if synth_defect_num > 0:
            defect_type_num = synth_defect_num // 4 # there are 4 different defect types. we wil generate equal number of those
            label = self.synth_def_label # label of synthetic defects is -1 by default
            indices = (-1,-1) # since these patches are not extracted from a real image, there are no corresponding indices  
            if noise_type == 'statistical':           
                for i in range(defect_type_num*4):
                    index = np.random.randint(0, len(self.samples) - 1)
                    img, _, _ = self.samples[index]
                    new_img = gaussian_noise_defect(img)
                    self.samples.append((new_img, label, indices))
                    cv.imwrite('E:/temp/samples/def/synth/ndef_'+str(i)+'_gauss.png', new_img)
            else:
                for i in range(defect_type_num):
                    index = np.random.randint(0, len(self.samples) - 1)
                    img, _, _ = self.samples[index]
                    ## ------------------
                    ## Use perlin noise
                    ## ------------------
                    #new_img_1 = horizontal_defect(img, im_size = self.patch_size, thickness = 8)              
                    #new_img_2 = vertical_defect(img, im_size = self.patch_size, thickness = 8)
                    #new_img_5 = spot_defect(img, im_size = self.patch_size, thickness = 30)               
                    #new_img_4 = spot_defect(img, im_size = self.patch_size, thickness = 6)  # spot defect function can generate big patch defects as well
                    ## ------------------
                    ## Use JuliaSet noise
                    ## ------------------
                    new_img_1 = horizontal_defect_julia_set(img, im_size = self.patch_size)               
                    new_img_2 = vertical_defect_julia_set(img, im_size = self.patch_size)                
                    new_img_3 = point_defect_julia_set(img, im_size = self.patch_size)
                    new_img_4 = point_defect_julia_set(img, im_size = self.patch_size)
                   
                    self.samples.append((new_img_1, label, indices))
                    self.samples.append((new_img_2, label, indices))
                    self.samples.append((new_img_3, label, indices))
                    self.samples.append((new_img_4, label, indices))
                    #cv.imwrite('E:/temp/samples/def/synth/ndef_'+str(i)+'_hr.jpg', new_img_1)
                    #cv.imwrite('E:/temp/samples/def/synth/ndef_'+str(i)+'_vr.jpg', new_img_2)
                    #cv.imwrite('E:/temp/samples/def/synth/ndef_'+str(i)+'_sp.jpg', new_img_3)
                    #cv.imwrite('E:/temp/samples/def/synth/ndef_'+str(i)+'_pc.jpg', new_img_4)

            logger.info("generated {} synthetic defected patches for a ratio of {}%".format(defect_type_num, self.defect_ratio))
            logger.info("Total number of training samples for this dataset now is {}".format(len(self.samples)))
        else:
            logger.info("requested {} defect number. Synthetic defects wil NOT BE generated...".format(len(self.samples)))
       

    def split_data(self, defect_samples, non_defect_samples):
        train_sample_count = int(len(non_defect_samples) * 0.7)
        test_normal_count = len(non_defect_samples) - train_sample_count
        test_defect_count = len(defect_samples)
        if self.mode == 'train':
            train_samples = []
            for i in range(train_sample_count):
                train_samples.append(non_defect_samples.pop())
            return train_samples
        else:
            test_samples = defect_samples
            # count from biggest index down to train_sample_count
            # this is to ensure that the training and test samples are from disjoint sets (for non-defect samples)
            for j in range(len(non_defect_samples),train_sample_count,-1):
                test_samples.append(non_defect_samples[j-1])
            return test_samples
    def get_population_stats(self):
        return self.sample_mean, self.sample_stddev

    def get_all_images(self):
        images = []
        for i in range(len(self.samples)):
            if self.transform is not None:
                images.append(self.transform(self.samples[i][0]))
            else:
                images.append(self.samples[i][0])

        return torch.stack(images)

    def get_all_labels(self):
        labels = []
        for i in range(len(self.samples)):
            label = self.samples[i][1]
            if self.target_transform is not None:                
                if label != -1:
                    label = self.target_transform(label)
            labels.append(label)

        return labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """Override the original method of the CIFAR10 class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
       
        img, target, indices_tuple = self.samples[index]
      
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            if target != -1:
                target = self.target_transform(target)

        return img, target, index

class AITEX_OnlineImgDataset(TorchvisionDataset):
    """
    This class holds the datasets for training and test.
    It uses the OnlineImagePatchFeeder class to generate the samples for both training and test.
    """
    def __init__(self, root: str, dataset_code = 'NoDatasetCode', normal_class = 1, train = False, gen_synthetic_defect = False, gen_validation_set = False, defect_ratio = 5, noise_type = 'structured', patch_size = PatchSize(32,32), batch_size = 64, img_color = 'COLOR'):
        super().__init__(root, dataset_code = dataset_code)
        assert img_color in ['COLOR','GRAY_SCALE']

        self.train = train
        self.batch_size = batch_size
        self.gen_validation_set = gen_validation_set
        if (root):
            self.img_root  = root
        else:
            self.img_root = 'E:/Anomaly Detection/AITEX IMAGES/original_samples'

        self.n_classes = 2  # 1: normal, 0: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 2))
        self.outlier_classes.remove(normal_class)
        # find the population mean and standard deviation
        #dataset = OnlineImagePatchFeeder(root=self.img_root, transform=None, target_transform=None, max_normal_sample_num = 800,
        #                                 mode = 'train', patch_size = patch_size, collect_stats = True)
        #mean, std_dev = dataset.get_population_stats() 
        
        transform = transforms.Compose([
            #transforms.Resize(32),
            
            transforms.ToTensor()
            #transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l2')),
            #transforms.Normalize(mean=mean, std=std_dev)
        ])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        self.train_set = OnlineImagePatchFeeder(root=self.img_root, transform=transform, target_transform=target_transform, mode = 'train', max_normal_sample_num = 1600, 
                                                gen_synthetic_defect = gen_synthetic_defect, synth_def_label = -1, defect_ratio = defect_ratio, noise_type = noise_type, patch_size = patch_size, img_color = img_color)
        self.test_set = OnlineImagePatchFeeder(root=self.img_root, transform=transform, target_transform=target_transform, mode = 'test', max_normal_sample_num = 1200, 
                                               gen_synthetic_defect = gen_synthetic_defect, defect_ratio = defect_ratio, noise_type = noise_type, patch_size = patch_size, img_color = img_color)

        if gen_validation_set:
            self.validation_set = OnlineImagePatchFeeder(root=self.img_root, transform=transform, target_transform=target_transform, mode = 'train', max_normal_sample_num = 500, 
                                                gen_synthetic_defect = gen_synthetic_defect, synth_def_label = 0, defect_ratio = defect_ratio, noise_type = noise_type, patch_size = patch_size, img_color = img_color)
        else:
            self.validation_set = None

    
           

if __name__ == '__main__':
    root = 'E:/Anomaly Detection/AITEX IMAGES/original_samples'
    fabric = 'fabric_01'
    dataset_root = root + os.sep + fabric
    dataset = AITEX_OnlineImgDataset(dataset_root)
    img_list = []
    for i in range(len(dataset.train_set)):
        X, _, _ = dataset.train_set[i]
        img_list.append(X)
    
    images, testsm_lbls, indices = zip(*dataset.test_set.samples)
    t_images= torch.tensor(images)
    num_defect =  testsm_lbls.count(0)
    num_normal =  testsm_lbls.count(1)
    train_loader, test_loader = dataset.loaders(batch_size = 64)
    for imgs, lbls, idxs in test_loader:
        an_img = imgs[0]
        m = tensor_to_img(an_img)
        import pylab as plt

        # show a sample image for preview
        imno = 1
        fig = plt.figure()
        plt.subplot(1,2, imno)
        plt.imshow(m)
        plt.colorbar()
        plt.title("image")
        plt.show()
        print(lbls, idxs)
        break

