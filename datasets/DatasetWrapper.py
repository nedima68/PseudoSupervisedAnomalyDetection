from .AITEXDataset import AITEX_Dataset
from .OnlineAITEXImageDataset import AITEX_OnlineImgDataset
from .Cifar10Dataset import CIFAR10_Dataset
from .MNISTDataset import MNIST_Dataset



def load_dataset(dataset_name, data_path, normal_class, gen_synthetic_defect = False, gen_validation_set = False, defect_ratio = 5, noise_type = 'structured', img_color = 'COLOR', dataset_code = 'NoDatasetCode'):
    """Loads the dataset."""

    implemented_datasets = ('AITEX', 'CIFAR10', 'MNIST')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'AITEX':
        #dataset = AITEX_Dataset(root=data_path, normal_class=normal_class)
        dataset = AITEX_OnlineImgDataset(root=data_path, normal_class=normal_class, dataset_code = dataset_code, gen_synthetic_defect = gen_synthetic_defect, gen_validation_set = gen_validation_set, defect_ratio = defect_ratio, noise_type = noise_type, img_color = img_color)

    if dataset_name == 'CIFAR10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'MNIST':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    return dataset
