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
GRAPH_NAME ='Rnn_4x'
RECURRENT = True
PARAMETERS = {
    'd1' : 'Rnn', 
    'd2' : 'Rnn', 
    'd3' : 'Rnn', 
    'b_' : 'Rnn', 
    'u1' : 'False', 
    'u2' : 'False', 
    'u3' : 'False'
}

# arguments
TIMESTEPS = 3
BATCH_SIZE = 32
NUM_EPOCHS = 50
INPUT_SIZE = 128
INPUT_CHANNELS = 3
NUM_CLASSES = 2
LEARNING_RATE = 0.001

# decive
DEVICE = "cuda:0"
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
device