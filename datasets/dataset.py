import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
from skimage import io, transform
from torchvision.transforms import ToPILImage, Compose, Resize, ToTensor


IS_FLOYD_ENV = True if os.getcwd() == '/floyd/home' else False


class ChexpertDataset(Dataset):
    """ Chexpert dataset """

    classes = ['0 - negative', 'u - uncertain', '1 - positive']

    classes_to_idx = {-1: 0, 0: 1, 1: 2}
    idx_to_classes = {2: 1, 1: 0, 0: -1}

    def __init__(self, data_dir, column_name, csv_file):
        """
        Args:
            csv_file (string): File containing paths and labels
            column_name (string): Treat this column as labels
        """
        # IS_FLOYD_ENV = True if os.getcwd() == '/floyd/home' else False
        self.root_dir = 'chexpert' if IS_FLOYD_ENV else 'CheXpert-v1.0-small/'
        self.data_dir = '/floyd/input' if IS_FLOYD_ENV else data_dir

        self.csv_file = csv_file
        

        cwd = '' if IS_FLOYD_ENV else os.getcwd().replace('chexpert', '')
        print(cwd)
        self.csv_path = os.path.join(cwd, self.data_dir, self.root_dir, self.csv_file)

        self.meta_data = pd.read_csv(self.csv_path)
        self.paths = self.meta_data['Path']

        # treat blanks as negative label, since nothing is mentioned
        # about a disease it can be taken as a negative label

        # preprocessing the labels
        labels = np.copy(self.meta_data[column_name])
        labels[np.isnan(labels)] = 0.0
        self.labels = labels
        self.transform = Compose([ToPILImage(), Resize((160, 160)), ToTensor()])

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.meta_data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        cwd = os.getcwd() if IS_FLOYD_ENV else os.getcwd().replace('chexpert', '')
        data_path = os.path.join(cwd, 'data', self.paths[index])

        if IS_FLOYD_ENV:
            data_path = data_path.replace('home/data/CheXpert-v1.0-small', 'input/chexpert')

        # Load data and get label
        X = io.imread(data_path)
        X.resize((*X.shape, 1))
        y = self.classes_to_idx[int(self.labels[index])]
        return self.transform(X), y
