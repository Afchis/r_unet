import torch
import torch.nn as nn

from args import *
from model_paths import *


'''
Model head
'''
class UNetDesigner(nn.Module):    
    def __init__(self, d1, d2, d3, b_, u1, u2, u3,
                 input_size=INPUT_SIZE, input_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES):
        super(UNetDesigner, self).__init__()
        self.num_classes = NUM_CLASSES
        self.d1, self.d2, self.d3, self.b, self.u1, self.u2, self.u3 = d1, d2, d3, b_, u1, u2, u3
        self.input_size = input_size
        self.input_chennels = input_channels
        self.ch_list = [self.input_chennels, 32, 64, 128, 256]
        self.input_x2 = int(self.input_size / 2)
        self.input_x4 = int(self.input_size / 4)
        self.input_x8 = int(self.input_size / 8)

         ##### Down_1 layer ##### input_size = 128
        if self.d1 == False:
            self.down1 = nn.Sequential(ConvRelu(self.ch_list[0], self.ch_list[1]),
                                       ConvRelu(self.ch_list[1], self.ch_list[1])
                                       )
                                                                                        # 1  -->32
        else:                                                                           # 32 -->32
            self.down1 = nn.Sequential(ConvRnnRelu(self.ch_list[0], self.ch_list[1],    # Channels
                                                   self.input_size, self.d1),               
                                       ConvRelu(self.ch_list[1], self.ch_list[1])                   
                                       )
        self.down1_pool = MaxPool()

         ##### Down_2 layer ##### input_size = 64
        if self.d2 == False:
            self.down2 = nn.Sequential(ConvRelu(self.ch_list[1], self.ch_list[2]),
                                       ConvRelu(self.ch_list[2], self.ch_list[2])
                                       )
                                                                                        # 32 -->64
        else:                                                                           # 64 -->64
            self.down2 = nn.Sequential(ConvRnnRelu(self.ch_list[1], self.ch_list[2],
                                                   self.input_x2, self.d2),
                                       ConvRelu(self.ch_list[2], self.ch_list[2])
                                       )
        self.down2_pool = MaxPool()

         ##### Down_3 layer ##### input_size = 32
        if self.d3 == False:
            self.down3 = nn.Sequential(ConvRelu(self.ch_list[2], self.ch_list[3]),
                                       ConvRelu(self.ch_list[3], self.ch_list[3])
                                       )
                                                                                        # 64 -->128
        else:                                                                           # 128-->128
            self.down3 = nn.Sequential(ConvRnnRelu(self.ch_list[2], self.ch_list[3], 
                                                   self.input_x4, self.d3),
                                       ConvRelu(self.ch_list[3], self.ch_list[3])
                                       )
        self.down3_pool = MaxPool()

         ##### Bottom layer ##### input_size = 16
        if self.b == False:
            self.bottom = nn.Sequential(ConvRelu(self.ch_list[3], self.ch_list[4]),
                                        ConvRelu(self.ch_list[4], self.ch_list[4])
                                        )
                                                                                        # 128-->256
        else:                                                                           # 256-->256
            self.bottom = nn.Sequential(ConvRnnRelu(self.ch_list[3], self.ch_list[4], 
                                                    self.input_x8, self.b),
                                        ConvRelu(self.ch_list[4], self.ch_list[4])
                                        )

         ##### Up_3 layer #####
        self.up_cat_3 = UpAndCat()
        if self.u3 == False:
            self.up_conv_3 = nn.Sequential(ConvRelu(self.ch_list[4]+self.ch_list[3], 
                                                    self.ch_list[3]),
                                           ConvRelu(self.ch_list[3], self.ch_list[3])
                                           )
                                                                                        # 394-->128
        else:                                                                           # 128-->128
            self.up_conv_3 = nn.Sequential(ConvRnnRelu(self.ch_list[4]+self.ch_list[3], 
                                                       self.ch_list[3], 
                                                       self.input_x4, self.u3),
                                           ConvRelu(self.ch_list[3], self.ch_list[3])
                                           )   

         ##### Up_2 layer #####
        self.up_cat_2 = UpAndCat()
        if self.u2 == False:
            self.up_conv_2 = nn.Sequential(ConvRelu(self.ch_list[3]+self.ch_list[2], 
                                                    self.ch_list[2]),
                                           ConvRelu(self.ch_list[2], self.ch_list[2])
                                           )
                                                                                        # 192-->64
        else:                                                                           # 64 -->64
            self.up_conv_2 = nn.Sequential(ConvRnnRelu(self.ch_list[3]+self.ch_list[2], 
                                                       self.ch_list[2], 
                                                       self.input_x2, self.u2),
                                           ConvRelu(self.ch_list[2], self.ch_list[2])
                                           )

         ##### Up_1 layer #####
        self.up_cat_1 = UpAndCat()
        if self.u1 == False:
            self.up_conv_1 = nn.Sequential(ConvRelu(self.ch_list[2]+self.ch_list[1], 
                                                    self.ch_list[1]),
                                           ConvRelu(self.ch_list[1], self.ch_list[1])
                                           )
                                                                                        # 96 -->32
        else:                                                                           # 32 -->32
            self.up_conv_1 = nn.Sequential(ConvRnnRelu(self.ch_list[2]+self.ch_list[1], 
                                                       self.ch_list[1], 
                                                       self.input_size, self.u1),
                                           ConvRelu(self.ch_list[1], self.ch_list[1])
                                           )

         ##### Final layer #####
        self.final = nn.Sequential(nn.Conv2d(self.ch_list[1], self.num_classes, kernel_size=1)
                                   )                                                    # 32-->NUM_CLASSES

    def forward(self, x):
        x = x.reshape(-1, self.input_chennels, self.input_size, self.input_size)
        # print(x.shape)
        down1_feat = self.down1(x)
        pool1 = self.down1_pool(down1_feat)
        # print(pool1.shape)
        down2_feat = self.down2(pool1)
        pool2 = self.down2_pool(down2_feat)
        # print(pool2.shape)
        down3_feat = self.down3(pool2)
        pool3 = self.down3_pool(down3_feat)
        # print(pool3.shape)
        bottom_feat = self.bottom(pool3)
        # print(bottom_feat.shape)
        up_feat3 = self.up_cat_3(bottom_feat, down3_feat)
        up_feat3 = self.up_conv_3(up_feat3)
        
        up_feat2 = self.up_cat_2(up_feat3, down2_feat)
        up_feat2 = self.up_conv_2(up_feat2)
        
        up_feat1 = self.up_cat_1(up_feat2, down1_feat)
        up_feat1 = self.up_conv_1(up_feat1)
        
        out = self.final(up_feat1)
        return out