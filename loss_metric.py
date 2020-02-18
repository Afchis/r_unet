from args import *

import torch
import torch.nn as nn
import torch.nn.functional as F


'''
Loss
'''
def l2_loss(x, y, d):
    y = y.reshape(-1, NUM_CLASSES, INPUT_SIZE, INPUT_SIZE)
    d = d.reshape(-1, 1, INPUT_SIZE, INPUT_SIZE)
    x = torch.sigmoid(x)
    out = (d*(x - y)**2).sum()
    #print(out.item())
    return out

def bce_loss(x, y, d):
    y = y.reshape(x.shape)
    return F.binary_cross_entropy_with_logits(x, y)

def dice_loss(x, y, d, smooth = 1.):
    y = y.reshape(-1, NUM_CLASSES, INPUT_SIZE, INPUT_SIZE)
    d = d.reshape(-1, 1, INPUT_SIZE, INPUT_SIZE)
    x = torch.sigmoid(x)
    intersection = (x * y).sum(dim=2).sum(dim=2)
    x_sum = x.sum(dim=2).sum(dim=2)
    y_sum = y.sum(dim=2).sum(dim=2)
    dice_loss = 1 - ((2*intersection + smooth) / (x_sum + y_sum + smooth))
    #print(dice_loss.mean().item())
    return dice_loss.mean()

def dice_combo_loss(x, y, d, bce_weight=0.5):
    dice_combo_loss = bce_weight * bce_loss(x, y, d) + (1 - bce_weight) * dice_loss(x, y, d)
    return dice_combo_loss

def l2_combo_loss(x, y, d):
    l2_combo_loss = l2_loss(x, y, d) * bce_loss(x, y, d)
    return l2_combo_loss


'''
Metric
'''
def IoU_metric(x, y ,smooth = 1.):
    y = y.reshape(x.shape)
    x = torch.sigmoid(x)
    y = torch.tensor((y > 0.5).float())
    intersection = (x * y).sum(dim=2).sum(dim=2)
    x_sum = x.sum(dim=2).sum(dim=2)
    y_sum = y.sum(dim=2).sum(dim=2)
    IoU_metric = (intersection + smooth) / (x_sum + y_sum - intersection + smooth)
    return IoU_metric.mean()