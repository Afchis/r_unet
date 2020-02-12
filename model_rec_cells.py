import torch
import torch.nn as nn

from args import *


'''
Recurrent cells
'''
class ConvRnnCell(nn.Module):    
    def __init__(self, in_channels, out_channels):
        super(ConvRnnCell, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels+out_channels, out_channels, kernel_size=3, padding=1))
             
    def forward(self, x, hidden):
        out = torch.cat([x, hidden],dim=1)
        out = self.conv1(out)
        hidden = out
        return out, hidden


class ConvGruCell(nn.Module):    
    def __init__(self, in_channels, out_channels):
        super(ConvGruCell, self).__init__()
        self.conv_for_input = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        
        self.conv_for_hidden = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        
        self.conv_2x_update = nn.Sequential(nn.Conv2d(in_channels+out_channels, out_channels, kernel_size=3, padding=1))
        self.conv_2x_reset = nn.Sequential(nn.Conv2d(in_channels+out_channels, out_channels, kernel_size=3, padding=1))
        
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

     
    def forward(self, x, hidden):
        input = torch.cat([x, hidden],dim=1)

        update_gate = self.conv_2x_update(input)
        update_gate = self.sig((update_gate)) ### output after update gate
        reset_gate = self.conv_2x_reset(input)
        reset_gate = self.sig((reset_gate)) ### output after reset gate
        
        
        memory_for_input = self.conv_for_input(x)
        memory_for_hidden = self.conv_for_hidden(hidden)# просто хидден

        memory_content = memory_for_input + (reset_gate * memory_for_hidden) ### output for reset gate(affects how the reset gate do work)
        memory_content = self.relu(memory_content)

        hidden = (update_gate * hidden) + ((1 - update_gate) * memory_content) # torch.ones(input_size, hidden_size)

        return hidden, hidden


class ConvRrnCell(nn.Module):    
    def __init__(self, in_channels, out_channels):
        super(ConvRrnCell, self).__init__()
        self.conv_for_input = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        
        self.conv_for_hidden = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        
        self.conv_2x_update = nn.Sequential(nn.Conv2d(in_channels+out_channels, out_channels, kernel_size=3, padding=1))
        self.conv_2x_reset = nn.Sequential(nn.Conv2d(in_channels+out_channels, out_channels, kernel_size=3, padding=1))
        
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

     
    def forward(self, x, hidden):
        input = torch.cat([x, hidden],dim=1)

        update_gate = self.conv_2x_update(input)
        update_gate = self.sig((update_gate)) 
        
        memory_for_input = self.conv_for_input(x)
        memory_for_hidden = hidden

        memory_content = memory_for_input + memory_for_hidden
        memory_content = self.relu(memory_content)

        hidden = (update_gate * hidden) + ((1 - update_gate) * memory_content) # torch.ones(input_size, hidden_size)

        return hidden, hidden


class ConvSruCell(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvSruCell, self).__init__()
        self.update_gate = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.backbone = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, hidden):
        z = self.update_gate(x)
        z = self.sig(z)
        
        h_ = self.backbone(x)
        h_ = self.tanh(h_)
        
        h_prev = hidden * z
        h = (1 - z) * h_
        out = h + h_prev
        return out, out


class ConvDruCell(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvDruCell, self).__init__()
        self.update_gate = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.reset_gate = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.backbone = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, hidden):
        z = self.update_gate(x)
        z = self.sig(z)
        
        r = self.reset_gate(x)
        r = self.sig(r)
        rx = r * x
        h_ = self.backbone(rx)
        h_ = self.tanh(h_)
        
        h_prev = hidden * z
        h = (1 - z) * h_
        out = h + h_prev
        return out, out


class ConvLstmCell(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvLstmCell, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels+out_channels, 4*out_channels, kernel_size=3, padding=1))
        self.hidden_dim = out_channels

    def forward(self, x, hidden):
        h_h, c_h = hidden
        
        combined = torch.cat([x, h_h], dim=1)
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_h + i * g
        h_next = o * torch.tanh(c_next)
        hidden_next = h_next, c_next
        
        return h_next, hidden_next