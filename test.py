from args import *
from model_head import *
from dataloader_COCO2017 import *

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from PIL import ImageSequence


to_pil = transforms.ToPILImage()

model = UNetDesigner(d1=PARAMETERS['d1'],
                     d2=PARAMETERS['d2'],
                     d3=PARAMETERS['d3'],
                     d4=PARAMETERS['d4'],
                     b_=PARAMETERS['b_'],
                     u4=PARAMETERS['u4'],
                     u3=PARAMETERS['u3'],
                     u2=PARAMETERS['u2'],
                     u1=PARAMETERS['u1']
                     )
model = model.to(device)

model.load_state_dict(torch.load('weights/weights.pth'))
'''
Test
'''
list_inp = []
list_out = []
for i, data in enumerate(test_loader):
    if i < 50:
        input, _, _ = data
        input = input.to(device)
        test_output = model(input)
        list_inp.append(input)
        list_out.append(test_output)
    else:
        break

def showw(object, i):
    imgs = object[i].cpu()
    img = torch.sigmoid(imgs[2][0])
    img = (img > 0.6).float()
    return to_pil(img)

index = 3 
for i in range(len(list_out)):
    test_out = showw(list_out, i)
    test_out.save("../r_unet/data/test_output/frame%d.png" % index)
    index += 1