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
GRAPH_NAME ='Lstm'
RECURRENT = True
PARAMETERS = {
    'd1' : 'False', 
    'd2' : 'False', 
    'd3' : 'False', 
    'b_' : 'Lstm', 
    'u1' : 'False', 
    'u2' : 'False', 
    'u3' : 'False'
}

# arguments
TIMESTEPS = 3
BATCH_SIZE = 16
NUM_EPOCHS = 15
INPUT_SIZE = 128
INPUT_CHANNELS = 3
NUM_CLASSES = 22
LEARNING_RATE = 0.0005

# decive
DEVICE = "cuda:0"
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
device