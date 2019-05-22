from torchvision import datasets, transforms
from base import BaseDataLoader
import datasets.dataset as custom_datasets


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(
            self,
            data_dir,
            batch_size,
            shuffle=True,
            validation_split=0.0,
            num_workers=1,
            training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(
            self.data_dir,
            train=training,
            download=True,
            transform=trsfm)
        super(
            MnistDataLoader,
            self).__init__(
            self.dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers)


class ChexpertDataLoader(BaseDataLoader):
    """
    CheXpert data loader
    """

    def __init__(
            self,
            data_dir,
            batch_size,
            column_name,
            shuffle=True,
            validation_split=0.0,
            num_workers=1,
            training=True):
        self.data_dir = data_dir
        self.dataset = custom_datasets.ChexpertDataset(data_dir, column_name)
        super(
            ChexpertDataLoader,
            self).__init__(
            self.dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers)
