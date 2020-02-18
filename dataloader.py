from args import *

import os
import numpy as np

from PIL import Image
import scipy.ndimage.morphology as morph

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms


# way to the data folders
FOLDER_DATA = "../r_unet/data/images"
FOLDER_MASK = "../r_unet/data/labels"
FOLDER_TEST = "../r_unet/data/test"
FOLDER_DATA_VAL = "../r_unet/data/images_val"
FOLDER_MASK_VAL = "../r_unet/data/labels_val"

FILE_NAMES = sorted(os.listdir('../r_unet/data/images'))
FILE_NAMES_VAL = sorted(os.listdir('../r_unet/data/images_val'))

# transforms
transform = transforms.Compose([
                              transforms.Resize((INPUT_SIZE, INPUT_SIZE), interpolation = 0),
                              transforms.ToTensor()
                              ])

to_tensor = transforms.ToTensor()


'''
Dataloader
'''
def get_labels(object):
    label1 = (object==0).float()
    depth1 = to_tensor(morph.distance_transform_edt(np.asarray(label1[0])))
    label2 = (label1==0).float()
    depth2 = to_tensor(morph.distance_transform_edt(np.asarray(label2[0])))
    depths = depth1 + depth2
    labels = torch.stack([label1, label2], dim=1).squeeze()
    return labels, depths


class TrainMedData(Dataset):
    def __init__(self):
        super().__init__()
        self.time = TIMESTEPS
        self.folder_data = FOLDER_DATA
        self.folder_mask = FOLDER_MASK
        self.file_names = FILE_NAMES

    def __getitem__(self, idx):
        gif_list = []
        gif_list_depth = []
        for i in range(self.time):
            gif_list.append(transform(Image.open(self.folder_data + '/' + self.file_names[idx+i])))
        gif_data = torch.stack(gif_list)
        gif_list.clear()
        for i in range(self.time):
            label, depth = get_labels(transform(Image.open(self.folder_mask + '/' + self.file_names[idx+i])))
            gif_list.append(label)
            gif_list_depth.append(depth)
        gif_mask = torch.stack(gif_list)
        gif_depth = torch.stack(gif_list_depth)
        gif_list.clear()
        gif_list_depth.clear()
        return gif_data, gif_mask, gif_depth

    def __len__(self):
        return len(self.file_names) - self.time + 1


class ValMedData(Dataset):
    def __init__(self):
        super().__init__()
        self.time = TIMESTEPS
        self.folder_data = FOLDER_DATA_VAL
        self.folder_mask = FOLDER_MASK_VAL
        self.file_names = FILE_NAMES_VAL

    def __getitem__(self, idx):
        gif_list = []
        gif_list_depth = []
        for i in range(self.time):
            gif_list.append(transform(Image.open(self.folder_data + '/' + self.file_names[idx+i])))
        gif_data = torch.stack(gif_list)
        gif_list.clear()
        for i in range(self.time):
            label, depth = get_labels(transform(Image.open(self.folder_mask + '/' + self.file_names[idx+i])))
            gif_list.append(label)
            gif_list_depth.append(depth)
        gif_mask = torch.stack(gif_list)
        gif_depth = torch.stack(gif_list_depth)
        gif_list.clear()
        gif_list_depth.clear()
        return gif_data, gif_mask, gif_depth

    def __len__(self):
        return len(self.file_names) - self.time + 1


class TestMedData(Dataset):
    def __init__(self):
        super().__init__
        self.time = TIMESTEPS
        self.folder_test = FOLDER_TEST
        self.file_names = FILE_NAMES + FILE_NAMES_VAL

    def __getitem__(self, idx):
        gif_list = []
        for i in range(self.time):
            gif_list.append(transform(Image.open(self.folder_test + '/' + self.file_names[idx+i])))
        gif_test = torch.stack(gif_list)
        gif_list.clear()
        return gif_test

    def __len__(self):
        return len(self.file_names) - self.time + 1


train_dataset = TrainMedData()
valid_dataset = ValMedData()
test_dataset = TestMedData()

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=1,
                          shuffle=False)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=1,
                          shuffle=False)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=1,
                         num_workers=1,
                         shuffle=False)

data_loaders = {
    'train' : train_loader,
    'valid' : valid_loader,
    'test' : test_loader
}

dataset_sizes = {
    'train': len(train_dataset),
    'valid': len(valid_dataset),
    'test': len(test_dataset)
}