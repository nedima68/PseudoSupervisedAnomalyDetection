

import torch
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
#from ImgProcessing.NCutFeatureExtractor import generate_NCUT_segmented_image
#from ImgProcessing.FeatureContourGenerator import draw_feature_contours
from datasets.preprocessing import get_target_label_idx, global_contrast_normalization, calculate_std_mean
import cv2
from PIL import Image
import json
import logging
import sys

# import matplotlib
# matplotlib.use('Agg')  # or 'PS', 'PDF', 'SVG'
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

global_counter = 0

def im_show(name, image, resize=1):
    H,W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize*W), round(resize*H))


def tensor_to_img(img, mean=0, std=1):    
    img = np.transpose(img.detach().cpu().numpy(), (1, 2, 0))
    img = (img*std+ mean)*255
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    return img

def normalize_to_zero_one_range(img):
    # assume torch tensor
    min = torch.min(img)
    max = torch.max(img)
    normalized_img = (img-min)/(max-min)
    return normalized_img

def save_img_patch(img, prefix = 'original'):
    global global_counter
    global_counter += 1
    cv2.imwrite('E:/temp/images/patches/patch_' + prefix + '_' + str(global_counter) + '.png', img)
    return img

def plot_images_grid(x: torch.tensor, export_img, title: str = '', nrow=6, padding=2, normalize=False, pad_value=0, save = True, apply_transforms = False):
    """Plot 4D Tensor of images of shape (B x C x H x W) as a grid."""
    logger = logging.getLogger()
    global global_counter
    #if apply_transforms:
    #    #global_counter = 0
        
    #    transform = transforms.Compose([
    #      #transforms.ToPILImage(),transforms.Lambda(lambda x: transforms.functional.adjust_brightness(x, brightness_factor = 1.1)),
    #      #transforms.Lambda(lambda x: normalize_to_zero_one_range(x)),
    #      transforms.Lambda(lambda x: tensor_to_img(x)),
    #      transforms.Lambda(lambda x: save_img_patch(x, prefix = 'original')),
    #      transforms.Lambda(lambda x: generate_NCUT_segmented_image(x)),
    #      transforms.Lambda(lambda x: save_img_patch(x, prefix = 'NCUT')),
    #      #transforms.Lambda(lambda x: draw_feature_contours(x)),
    #      transforms.ToTensor()         
    #    ])
        
    #    for i in range(x.shape[0]): 
    #        try:
    #            x[i] = transform(x[i])
    #        except:
    #           logger.error("Exception occurred while appliying transform {}".format( sys.exc_info()[0]))
    #           logger.error("Was processing image number {} with global counter {}".format(str(i), global_counter))

    grid = make_grid(x, nrow=nrow, padding=padding, normalize=normalize, pad_value=pad_value)
    npgrid = grid.detach().cpu().numpy()

    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')

    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if not (title == ''):
        plt.title(title)
    if (save == True):
        plt.savefig(export_img, bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()
    plt.clf()



def calculate_std_mean(img_root):
    
    data_transform = transforms.Compose([ transforms.ToTensor()])

    fabric_dataset = datasets.ImageFolder(root=img_root,
                                               transform=data_transform)
    dataset_loader = DataLoader(fabric_dataset, batch_size=4, shuffle=False,
                                                 num_workers=4)
    img_tensor = fabric_dataset[10][0]
    print(img_tensor)
    pop_mean = []
    pop_std0 = []
    pop_std1 = []
    count = 0
    for i, data in enumerate(dataset_loader, 0):
        # shape (batch_size, 3, height, width)
        numpy_image = data[0].numpy() 
        images, labels = data
        num = len(images)
        count += num
        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0,2,3))
        batch_std0 = np.std(numpy_image, axis=(0,2,3))
        batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)
        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)
        #pop_std1.append(batch_std1)

    print("total image processed = ", count)
    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std0 = np.array(pop_std0).mean(axis=0)
    #pop_std1 = np.array(pop_std1).mean(axis=0)

    return pop_mean, pop_std0

 


class MyCustomDS(datasets.ImageFolder):
    """Torchvision CIFAR10 class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, root, transform, *args, **kwargs):
        super(MyCustomDS, self).__init__(root, transform, *args, **kwargs)

    def __getitem__(self, index):
        """Override the original method of the CIFAR10 class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

def generate_samples(samples_indices, source_path):
    data_transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=pop_mean, std=pop_std)
        ])
    fabric_dataset = MyCustomDS(root=source_path, transform=data_transform)
    dataset_loader = DataLoader(fabric_dataset, batch_size=12, shuffle=False,
                                                 num_workers=4)
    
    samples = [fabric_dataset[i] for i in samples_indices]
    sample_imgs = []
    for i, tensor in enumerate(samples, 0):
        sample_imgs.append(tensor[0].tolist())

    return sample_imgs

def display_samples(samples_indices, source_path, target_path, save):
    if samples_indices == None:
        print('nothing to display ...')
        return

    sample_imgs =  generate_samples(samples_indices, source_path)
    plot_images_grid(torch.tensor(sample_imgs),target_path, save)


if __name__ == '__main__':
    display_samples(None, None, None)
