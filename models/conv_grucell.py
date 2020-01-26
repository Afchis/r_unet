import torch
import torch.nn as nn

# create class GruCell
class GruCell(nn.Module):
    
    def __init__(self, channel):
        super(GruCell, self).__init__()
        self.update_gate = nn.Conv2d(in_channels=channel+channel, out_channels=channel, kernel_size=3, stride=1, padding=1)
        
        self.reset_gate = nn.Conv2d(in_channels=channel+channel, out_channels=channel, kernel_size=3, stride=1, padding=1)
        
        self.memory_gate_for_input = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1)
        self.memory_gate_for_hidden = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1)
        
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()        

     
    def forward(self, x, hidden):
        input = torch.cat([x, hidden],dim=1)

        update_gate = self.update_gate(input)
        update_gate = self.sig((update_gate)) ### output after update gate
        
        reset_gate = self.reset_gate(input)
        reset_gate = self.sig((reset_gate)) ### output after reset gate
        
        memory_gate_for_input = self.memory_gate_for_input(x)
        memory_gate_for_hidden = self.memory_gate_for_hidden(hidden)

        memory_content = self.tanh((memory_gate_for_input + (reset_gate * memory_gate_for_hidden))) ### output for reset gate(affects how the reset gate do work)
        
        hidden = (update_gate * hidden) + ((1 - update_gate) * memory_content) # torch.ones(input_size, hidden_size)

        return hidden, hidden