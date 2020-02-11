import torch
import torch.nn as nn
import torch.nn.functional as F


'''
Loss
'''
def l2_loss(x, y, d):
    d = d.reshape(x.shape)
    y = y.reshape(x.shape)
    x = torch.sigmoid(x)
    out = ((x - y*d**2)**2).sum()
    return out

def bce_loss(x, y, d):
    y = y.reshape(x.shape)
    return F.binary_cross_entropy_with_logits(x, y)

def dice_loss(x, y, d):
    d = d.reshape(x.shape)
    y = y.reshape(x.shape)
    x = torch.sigmoid(x)
    intersection = (x * y*d**2).sum(dim=2).sum(dim=2)
    x_sum = (x*d**2).sum(dim=2).sum(dim=2)
    y_sum = (y*d**2).sum(dim=2).sum(dim=2)
    dice_loss = 1 - (2*intersection / (x_sum + y_sum))
    return dice_loss.mean()

def combo_loss(x, y, d, bce_weight=0.5):
    combo_loss = bce_weight * bce_loss(x, y, d) + (1 - bce_weight) * dice_loss(x, y, d)
    return combo_loss

def l2_combo_loss(x, y, d):
    l2_combo_loss = l2_loss(x, y, d) * bce_loss(x, y, d)
    return l2_combo_loss


'''
Metric
'''
def IoU_metric(x, y):
    y = y.reshape(x.shape)
    x = torch.sigmoid(x)
    intersection = (x * y).sum(dim=2).sum(dim=2)
    x_sum = x.sum(dim=2).sum(dim=2)
    y_sum = y.sum(dim=2).sum(dim=2)
    IoU_metric = intersection / (x_sum + y_sum - intersection)
    return IoU_metric.mean()