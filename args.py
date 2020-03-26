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
GRAPH_NAME ='False'
RECURRENT = True
PARAMETERS = {
    'd1' : 'False', 
    'd2' : 'False', 
    'd3' : 'False', 
    'd4' : 'False',
    'b_' : 'False', 
    'u4' : 'False', 
    'u3' : 'False', 
    'u2' : 'False',
    'u1' : 'False',
}

# arguments
TIMESTEPS = 1
BATCH_SIZE = 8
NUM_EPOCHS = 150
INPUT_SIZE = 256
INPUT_CHANNELS = 3
NUM_CLASSES = 2
LEARNING_RATE = 0.001

# decive
DEVICE = "cuda:1"
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
device



# # for Small_UNet
# PARAMETERS = {
#     'd1' : 'False', 
#     'd2' : 'False', 
#     'd3' : 'False', 
#     'b_' : 'False', 
#     'u3' : 'False', 
#     'u2' : 'False', 
#     'u1' : 'False',
# }

# # arguments
# TIMESTEPS = 1
# BATCH_SIZE = 16
# NUM_EPOCHS = 50
# INPUT_SIZE = 128
# INPUT_CHANNELS = 3
# NUM_CLASSES = 2
# LEARNING_RATE = 0.001