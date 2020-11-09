from .base_dataset import BaseADDataset
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


class TorchvisionDataset(BaseADDataset):
    """TorchvisionDataset class for datasets already implemented in torchvision.datasets."""

    def __init__(self, root: str, dataset_code):
        super().__init__(root, dataset_code)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)
        return train_loader, test_loader

    def get_validation_loader(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0):
        assert (self.validation_set != None), "ERROR: validation loader couldn't find a valid validation dataset, please make sure that you dataset implementation generates a proper validation dataset "

        validation_loader = DataLoader(dataset=self.validation_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
        return validation_loader

class CustomImageFolder(datasets.ImageFolder):
    """Torchvision CIFAR10 class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, root, transform, *args, **kwargs):
        super(CustomImageFolder, self).__init__(root, transform, *args, **kwargs)

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