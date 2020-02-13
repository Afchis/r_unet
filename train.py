import torch
from torch.utils.tensorboard import SummaryWriter

from args import *
from model_head import *
from dataloader_LITS import *
from loss_metric import *


writer = SummaryWriter()

model = UNetDesigner(d1=PARAMETERS['d1'],
                     d2=PARAMETERS['d2'],
                     d3=PARAMETERS['d3'],
                     b_=PARAMETERS['b_'],
                     u3=PARAMETERS['u3'],
                     u2=PARAMETERS['u2'],
                     u1=PARAMETERS['u1']
                     )
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

'''
Train
'''
val_metric = []
print('-'*30)
print('Tensorboard name: ', GRAPH_NAME)
for epoch in range(NUM_EPOCHS):# NUM_EPOCHS = 125
    print('*'*10, 'epoch: ', epoch, '*'*10)
    for phase in ['train', 'valid']:
        if phase == 'train':
            loss_list = []
            metric_list = []
            model.train()
            for i, data in enumerate(data_loaders[phase]):
                input, label, depth = data
                input = input.to(device)
                label = label.to(device)
                depth = depth.to(device)
                output = model(input)
                loss = bce_loss(output, label, depth)
                metric = IoU_metric(output, label)
                loss_list.append(loss.item())
                metric_list.append(metric.item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            train_mean_loss = sum(loss_list) / len(loss_list)
            train_mean_metric = sum(metric_list) / len(metric_list)
            print("train l2_norm: ", train_mean_metric)
        elif phase == 'valid':
            loss_list = []
            metric_list = []
            model.eval()
            for i, data in enumerate(data_loaders[phase]):
                input, label, depth = data
                input = input.to(device)
                label = label.to(device)
                depth = depth.to(device)
                output = model(input)
                loss = bce_loss(output, label, depth)
                metric = IoU_metric(output, label)
                loss_list.append(loss.item())
                metric_list.append(metric.item())
            valid_mean_loss = sum(loss_list) / len(loss_list)
            valid_mean_metric = sum(metric_list) / len(metric_list)
            print("val l2_norm: ", valid_mean_metric)
            val_metric.append(valid_mean_metric)
    writer.add_scalars('%s_loss' % GRAPH_NAME, {'train' : train_mean_loss, 
                                                'valid' : valid_mean_loss}, epoch)
    writer.add_scalars('%s_metric' % GRAPH_NAME, {'train' : train_mean_metric, 
                                                  'valid' : valid_mean_metric}, epoch)


print('Maximum Valid metric: ', max(val_metric))
print('Tensorboard name: ', GRAPH_NAME)
writer.close()
# !tensorboard --logdir=runs

#torch.save(model.state_dict(), 'weights/weights.pth')