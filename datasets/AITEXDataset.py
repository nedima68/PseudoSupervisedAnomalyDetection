from torch.utils.data import Subset
from PIL import Image
from torchvision import transforms, datasets
from base.torchvision_dataset import TorchvisionDataset, CustomImageFolder
from .preprocessing import get_target_label_idx, global_contrast_normalization, calculate_std_mean



class AITEX_Dataset(TorchvisionDataset):

    def __init__(self, root: str, dataset_code = 'NoDatasetCode', normal_class = 1, train = False, batch_size = 4):
        super().__init__(root, dataset_code = dataset_code)
        
        self.train = train
        self.batch_size = batch_size
        if (root):
            self.img_root  = root
        else:
            self.img_root = "E:/openCV/VStudioProjects/OpenCVExample-1/PyTorchBasic/FabricDefectDetection/images/AITEX/64x64"
        self.train_root = self.img_root + "/train";
        self.test_root = self.img_root + "/test";
        self.n_classes = 2  # 1: normal, 0: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 2))
        self.outlier_classes.remove(normal_class)
        # find the population mean and standard deviation
        # mean, std_dev = calculate_std_mean(self.train_root) 
        transform = transforms.Compose([
            #transforms.Resize(32),
            transforms.ToTensor()
            #transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
            #transforms.Normalize(mean=mean, std=std_dev)
        ])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        self.train_set = CustomImageFolder(root=self.train_root, transform=transform, target_transform=target_transform)
        self.test_set = CustomImageFolder(root=self.test_root, transform=transform, target_transform=target_transform)
           


