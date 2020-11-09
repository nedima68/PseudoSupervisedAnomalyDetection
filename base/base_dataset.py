from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np


class BaseADDataset(ABC):
    """ Anomaly detection dataset base class.
        
    """

    def __init__(self, root: str, dataset_code = 'NoDatasetCode'):
        super().__init__()
        self.root = root  # root path to data
        self.dataset_code = dataset_code
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = None  # tuple with original class labels that define the normal class
        self.outlier_classes = None  # tuple with original class labels that define the outlier class

        self.train_set = None  # must be of type torch.utils.data.Dataset
        self.test_set = None  # must be of type torch.utils.data.Dataset
        self.validation_set = None  # must be of type torch.utils.data.Dataset. Not all datasets is required to implement this. If implemented its loader should be retrieved using get_validation_loader() method

    @abstractmethod
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        """Implement data loaders of type torch.utils.data.DataLoader for train_set and test_set."""
        pass

    @abstractmethod
    def get_validation_loader(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> DataLoader:
        """
        Implement data loaders of type torch.utils.data.DataLoader for train_set and test_set.
        Added later to maintain backwards compatibility of existing code using the loaders() method
        """
        pass
        
    def __repr__(self):
        return self.__class__.__name__


class CustomGenericImageFeeder:
    """Torchvision class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, images, labels = None, transform = None, target_transform = None, mode = 'train', *args, **kwargs):
        assert mode in ['train','test'], "ERROR: invalid mode given ! must be one of 'train or 'test' "
        
        self.transform = transform
        self.target_transform = target_transform
        if labels == None:
            # if no labels given, generate '0' labels. 
            labels = [0] * len(images)
              
        self.samples = list(zip(images, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
       
        img, target = self.samples[index]
      
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            if target != -1:
                target = self.target_transform(target)

        return img, target, index