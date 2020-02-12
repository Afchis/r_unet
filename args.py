import torch


'''
Parameters
'''
"""
cell_model = {
    'Rnn' : ConvRnnCell(in_channels, out_channels), 
    'Gru' : ConvGruCell(in_channels, out_channels), 
    'Rrn' : ConvRrnCell(in_channels, out_channels), 
    'Sru' : ConvSruCell(in_channels, out_channels), 
    'Dru' : ConvDruCell(in_channels, out_channels),
    'Lstm': ConvLstmCell(in_channels, out_channels)
}
"""
RECURRENT = True
PARAMETERS = {
    'd1':'Rnn', 
    'd2':'Rnn', 
    'd3':'Rnn', 
    'b_':'Lstm', 
    'u1':'Rnn', 
    'u2':'Rnn', 
    'u3':'Rnn'
}

# arguments
TIMESTEPS = 3
BATCH_SIZE = 1
NUM_EPOCHS = 200
INPUT_SIZE = 128
INPUT_CHANNELS = 1
NUM_CLASSES = 2
LEARNING_RATE = 0.001

# decive
DEVICE = "cuda:1"
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
device