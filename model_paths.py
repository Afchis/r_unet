import torch
import torch.nn as nn

from args import *
from model_rec_cells import *


'''
Model paths
'''
class ConvRnn(nn.Module):
    def __init__(self, in_channels, out_channels, ConvRnn_input_size, cell_model, reccurent=RECURRENT): # arg for ConvRnn layer
        super(ConvRnn, self).__init__()
        self.cell_dict = {
            'Rnn' : ConvRnnCell(in_channels, out_channels), 
            'Gru' : ConvGruCell(in_channels, out_channels), 
            'Rrn' : ConvRrnCell(in_channels, out_channels), 
            'Sru' : ConvSruCell(in_channels, out_channels), 
            'Dru' : ConvDruCell(in_channels, out_channels)
        }
        self.rec = reccurent
        self.cell_model = cell_model
        self.batch_size = BATCH_SIZE
        self.timesteps = TIMESTEPS
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = ConvRnn_input_size
        self.hidden_size = (self.batch_size, self.out_channels, self.input_size, self.input_size)
        
        self.ConvRnn_layer = self.cell_dict[self.cell_model]
        self.init_hidden = torch.zeros(self.hidden_size).to(device)


    def forward(self, x):
        x_cells = None
        x_list = []

        x = x.reshape(self.batch_size, self.timesteps, self.in_channels, self.input_size, self.input_size)
        x = x.permute(1, 0, 2, 3, 4)
        if self.rec == True:
            for i in range(self.timesteps):
                if x_cells is None:
                    x_cells, hidden = self.ConvRnn_layer(x[i], self.init_hidden)
                    x_list.append(x_cells)
                else:
                    x_i, hidden = self.ConvRnn_layer(x[i], hidden)
                    x_list.append(x_i)
        elif self.rec == False:
            for i in range(self.timesteps):
                if x_cells is None:
                    x_cells, _ = self.ConvRnn_layer(x[i], self.init_hidden)
                    x_list.append(x_cells)
                else:
                    x_i, _ = self.ConvRnn_layer(x[i], self.init_hidden)
                    x_list.append(x_i)
        else:
            print('RECURRENT can be only True or False')
            quit()
        x_cells = torch.stack(x_list)
        x_cells = x_cells.permute(1, 0, 2, 3, 4)

        x_cells = x_cells.reshape(-1, self.out_channels, self.input_size, self.input_size)
        return x_cells


class ConvRnnRelu(nn.Module):
    def __init__(self, in_channels, out_channels ,ConvRnn_input_size, cell_model):
        super(ConvRnnRelu, self).__init__()
        self.cell_model = cell_model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = ConvRnn_input_size
        self.convrnnrelu = nn.Sequential(ConvRnn(self.in_channels, self.out_channels, 
                                                 self.input_size, self.cell_model),
                                         nn.ReLU()
                                         )

    def forward(self, x):
        return self.convrnnrelu(x)


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvRelu, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.convrelu = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
                                      nn.ReLU()
                                      )

    def forward(self, x):
        return self.convrelu(x)


class MaxPool(nn.Module):
    def __init__(self):
        super(MaxPool, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        return self.maxpool(x)


class UpAndCat(nn.Module):    
    def __init__(self):
        super(UpAndCat, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x_up, x_cat):
        out = self.up(x_up)
        out = torch.cat([out, x_cat], dim=1)
        return out