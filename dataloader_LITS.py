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
FOLDER_DATA = "/headless/afchi_projects/_data_/LITS/Training_Batch_1/volume/imgs"
FOLDER_MASK = "/headless/afchi_projects/_data_/LITS/Training_Batch_1/segmentation/imgs"
FOLDER_DATA_VAL = "/headless/afchi_projects/_data_/LITS/Training_Batch_1/volume/imgs_val"
FOLDER_MASK_VAL = "/headless/afchi_projects/_data_/LITS/Training_Batch_1/segmentation/imgs_val"
FOLDER_DATA_TEST = "/headless/afchi_projects/_data_/LITS/Training_Batch_1/volume/imgs_test"
FOLDER_MASK_TEST = "/headless/afchi_projects/_data_/LITS/Training_Batch_1/segmentation/imgs"

FOLDER_DATA_NAMES = sorted(os.listdir(FOLDER_DATA))
FOLDER_MASK_NAMES = sorted(os.listdir(FOLDER_MASK))
FOLDER_DATA_VAL_NAMES = sorted(os.listdir(FOLDER_DATA_VAL))
FOLDER_MASK_VAL_NAMES = sorted(os.listdir(FOLDER_MASK_VAL))
FOLDER_DATA_TEST_NAMES = sorted(os.listdir(FOLDER_DATA_TEST))
FOLDER_MASK_TEST_NAMES = sorted(os.listdir(FOLDER_MASK_TEST))

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

        self.folder_data_names = FOLDER_DATA_NAMES
        self.folder_mask_names = FOLDER_MASK_NAMES

    def __getitem__(self, idx):
        gif_list = []
        gif_list_depth = []
        for i in range(self.time):
            gif_list.append(transform(Image.open(self.folder_data + '/' + self.folder_data_names[idx+i])))
        gif_data = torch.stack(gif_list)
        gif_list.clear()
        for i in range(self.time):
            label, depth = get_labels(transform(Image.open(self.folder_mask + '/' + self.folder_mask_names[idx+i]).convert('L')))
            gif_list.append(label)
            gif_list_depth.append(depth)
        gif_mask = torch.stack(gif_list)
        gif_depth = torch.stack(gif_list_depth)
        gif_list.clear()
        gif_list_depth.clear()
        return gif_data, gif_mask, gif_depth

    def __len__(self):
        return len(self.folder_data_names) - self.time + 1


class ValMedData(Dataset):
    def __init__(self):
        super().__init__()
        self.time = TIMESTEPS
        self.folder_data = FOLDER_DATA_VAL
        self.folder_mask = FOLDER_MASK_VAL

        self.folder_data_names = FOLDER_DATA_VAL_NAMES
        self.folder_mask_names = FOLDER_MASK_VAL_NAMES

    def __getitem__(self, idx):
        gif_list = []
        gif_list_depth = []
        for i in range(self.time):
            gif_list.append(transform(Image.open(self.folder_data + '/' + self.folder_data_names[idx+i])))
        gif_data = torch.stack(gif_list)
        gif_list.clear()
        for i in range(self.time):
            label, depth = get_labels(transform(Image.open(self.folder_mask + '/' + self.folder_mask_names[idx+i]).convert('L')))
            gif_list.append(label)
            gif_list_depth.append(depth)
        gif_mask = torch.stack(gif_list)
        gif_depth = torch.stack(gif_list_depth)
        gif_list.clear()
        gif_list_depth.clear()
        return gif_data, gif_mask, gif_depth

    def __len__(self):
        return len(self.folder_data_names) - self.time + 1


class TestMedData(Dataset):
    def __init__(self):
        super().__init__()
        self.time = TIMESTEPS
        self.folder_data = FOLDER_DATA_TEST
        self.folder_mask = FOLDER_MASK_TEST

        self.folder_data_names = FOLDER_DATA_TEST_NAMES
        self.folder_mask_names = FOLDER_MASK_TEST_NAMES

    def __getitem__(self, idx):
        gif_list = []
        gif_list_depth = []
        for i in range(self.time):
            gif_list.append(transform(Image.open(self.folder_data + '/' + self.folder_data_names[idx+i])))
        gif_data = torch.stack(gif_list)
        gif_list.clear()
        for i in range(self.time):
            label, depth = get_labels(transform(Image.open(self.folder_mask + '/' + self.folder_mask_names[idx+i].convert('L'))))
            gif_list.append(label)
            gif_list_depth.append(depth)
        gif_mask = torch.stack(gif_list)
        gif_depth = torch.stack(gif_list_depth)
        gif_list.clear()
        gif_list_depth.clear()
        return gif_data, gif_mask, gif_depth

    def __len__(self):
        return len(self.folder_data_names) - self.time + 1


train_dataset = TrainMedData()
valid_dataset = ValMedData()
test_dataset = TestMedData()

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=2,
                          shuffle=True)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=2,
                          shuffle=True)

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