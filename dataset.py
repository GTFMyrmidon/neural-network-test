import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class CarsDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        '''
        Args:
            csv_file (str): Path to CSV file with annotations
            root_dir (str): Directory with all the images
            transform (callable, optional): Optional transform to be applied to a sample
        '''
        self.images = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images.ix[idx, 0])
        image = io.imread(img_name)
        labels = self.images.ix[idx, 1:].as_matrix().astype('float')
        labels = labels.reshape(-1, 2)
        sample = {'image': image, 'label': }
