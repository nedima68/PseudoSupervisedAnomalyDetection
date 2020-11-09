import torch
import numpy as np
from torchvision import transforms, datasets


def get_target_label_idx(labels, targets):
    """
    Get the indices of labels that are included in targets.
    :param labels: array of labels
    :param targets: list/tuple of target labels
    :return: list with indices of target labels
    """
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()


def normalize_data_bw_zero_one(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def global_contrast_normalization(x: torch.tensor, scale='l2'):
    """
    Apply global contrast normalization to tensor, i.e. subtract mean across features (pixels) and normalize by scale,
    which is either the standard deviation, L1- or L2-norm across features (pixels).
    Note this is a *per sample* normalization globally across features (and not across the dataset).
    """

    assert scale in ('l1', 'l2')

    n_features = int(np.prod(x.shape))

    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean

    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))

    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features

    x /= x_scale

    return x

def calculate_std_mean(img_root):
        
    data_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    fabric_dataset = datasets.ImageFolder(root=img_root,
                                                transform=data_transform)
    dataset_loader = torch.utils.data.DataLoader(fabric_dataset,
                                                    batch_size=4, shuffle=False,
                                                    num_workers=4)
    pop_mean = []
    pop_std0 = []
    pop_std1 = []
    for i, data in enumerate(dataset_loader, 0):
        # shape (batch_size, 3, height, width)
        numpy_image = data[0].numpy()       
        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0,2,3))
        batch_std0 = np.std(numpy_image, axis=(0,2,3))
        #batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)
        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)
        #pop_std1.append(batch_std1)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std0 = np.array(pop_std0).mean(axis=0)
    #pop_std1 = np.array(pop_std1).mean(axis=0)

    return pop_mean, pop_std0