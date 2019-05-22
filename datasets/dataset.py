import torch
import pandas as pd
import os
from torch.utils.data import Dataset
from skimage import io, transform
from torchvision.transforms import ToPILImage, Compose, Resize, ToTensor


class ChexpertDataset(Dataset):
    """ Chexpert dataset """
    def __init__(self, csv_file, column_name):
        """
        Args:
            csv_file (string): File containing paths and labels
            column_name (string): Treat this column as labels
        """
        self.meta_data = pd.read_csv(csv_file)
        self.paths = self.meta_data['Path']
        self.labels = self.meta_data[column_name]
        self.transform = Compose([ToPILImage(), Resize((320, 320)), ToTensor()])

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.meta_data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        data_path = os.path.join(os.getcwd(), 'data', self.paths[index])

        # Load data and get label
        X = io.imread(data_path)
        y = self.labels[index]

        return self.transform(X), y
