from conv_grucell import GruCell

import torch
import torch.nn as nn


# create class Gru
class Gru(nn.Module):

    def __init__(self, channels_size, input_size):
        super(Gru, self).__init__()
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.channels_size = channels_size
        self.input_size = input_size
        self.hidden_size = (self.batch_size, channels_size, input_size, input_size)
        
        self.gru_layer0 = GruCell(channels_size)
        self.init_hidden = torch.zeros(self.hidden_size).to(device)
        self.gru_nan = gru_nan


    def forward(self, x):
        x_cells = None
        x_list = []
        if self.gru_nan == False:
            try:
                x = x.reshape(self.batch_size, self.timesteps, self.channels_size, self.input_size, self.input_size)
                x = x.permute(1, 0, 2, 3, 4)
                for i in range(timesteps):
                    if x_cells is None:
                        x_cells, hidden = self.gru_layer0(x[i], self.init_hidden)
                        x_list.append(x_cells)
                    else:
                        x_i, hidden = self.gru_layer0(x[i], hidden)
                        x_list.append(x_i)
                x_cells = torch.stack(x_list)

             ##### FOR LAST BATCH
            except RuntimeError:
                x = x.reshape(1, self.timesteps, self.channels_size, self.input_size, self.input_size) #last batch is (15), but batch_size = 16, #arg.timesteps = 2 
                x = x.permute(1, 0, 2, 3, 4)
                hidden_zero = torch.zeros_like(x)
                for i in range(timesteps):
                    if x_cells is None:
                        x_cells, hidden = self.gru_layer0(x[i], hidden_zero[0])
                        x_list.append(x_cells)
                    else:
                        x_i, hidden = self.gru_layer0(x[i], hidden)
                        x_list.append(x_i)
                x_cells = torch.stack(x_list)
             #####
        elif self.gru_nan == True:
            try:
                x = x.reshape(self.batch_size, self.timesteps, self.channels_size, self.input_size, self.input_size)
                x = x.permute(1, 0, 2, 3, 4)
                for i in range(timesteps):
                    if x_cells is None:
                        x_cells, hidden = self.gru_layer0(x[i], self.init_hidden)
                        x_list.append(x_cells)
                    else:
                        x_i, hidden = self.gru_layer0(x[i], self.init_hidden)
                        x_list.append(x_i)
                x_cells = torch.stack((x_cells, x_i))
            ##### FOR LAST BATCH
            except RuntimeError:
                x = x.reshape(1, self.timesteps, self.channels_size, self.input_size, self.input_size) #last batch is (15), but batch_size = 16, #arg.timesteps = 2 
                x = x.permute(1, 0, 2, 3, 4)
                hidden_zero = torch.zeros_like(x)
                for i in range(timesteps):
                    if x_cells is None:
                        x_cells, hidden = self.gru_layer0(x[i], hidden_zero[0])
                        x_list.append(x_cells)
                    else:
                        x_i, hidden = self.gru_layer0(x[i], hidden)
                        x_list.append(x_i)
                x_cells = torch.stack((x_cells, x_i))
        else:
            print('gru_nan can be only True or False')
            quit()

        return x_cells  