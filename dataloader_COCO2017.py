from args import *
from data.cocostuff import CocoStuff164k

import os
import numpy as np

from PIL import Image
import scipy.ndimage.morphology as morph

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms


# way to the data folders
# FOLDER_DATA = "/storage/ProtopopovI/_data_/COCO/2014/train2014"
# FOLDER_MASK = "/storage/ProtopopovI/_data_/COCO/2014/mask_train_2014"
# # FOLDER_TEST = "../r_unet/data/test"
# FOLDER_DATA_VAL = "/storage/ProtopopovI/_data_/COCO/2014/val2014"
# FOLDER_MASK_VAL = "/storage/ProtopopovI/_data_/COCO/2014/mask_val_2014"

# FILE_NAMES = sorted(os.listdir("/storage/ProtopopovI/_data_/COCO/2014/mask_train_2014"))
# FILE_NAMES_VAL = sorted(os.listdir("/storage/ProtopopovI/_data_/COCO/2014/mask_val_2014"))

# # transforms
# transform = transforms.Compose([
#                               transforms.Resize((INPUT_SIZE, INPUT_SIZE), interpolation = 0),
#                               transforms.ToTensor()
#                               ])

# to_tensor = transforms.ToTensor()


'''
Dataloader
'''
train_dataset = CocoStuff164k(
    root="/storage/ProtopopovI/_data_/COCO/COCO",
    split="train2017",
    ignore_label=255,
    mean_bgr=(104.008, 116.669, 122.675),
    augment=True,
    crop_size=256,
    scales=(0.5, 0.75, 1.0, 1.25, 1.5),
    flip=True,
)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=4,
                          shuffle=True)

valid_dataset = CocoStuff164k(
    root="/storage/ProtopopovI/_data_/COCO/COCO",
    split="val2017",
    ignore_label=255,
    mean_bgr=(104.008, 116.669, 122.675),
    augment=True,
    crop_size=256,
    scales=(0.5, 0.75, 1.0, 1.25, 1.5),
    flip=True,
)

valid_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=4,
                          shuffle=True)


data_loaders = {
    'train' : train_loader,
    'valid' : valid_loader
    # 'test' : test_loader
}

dataset_sizes = {
    'train': len(train_dataset),
    'valid': len(valid_dataset)
    # 'test': len(test_dataset)
}
if __name__ == '__main__':
    img, mask, depth = train_dataset[0]
    print('img: ', img, 'mask: ', mask, 'depth: ', depth)
    print('img.shape: ', img.shape, 'mask.shape: ', mask.shape, 'depth: ', depth.shape)