import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from args import *
from model_head import *
from dataloader_COCO2014_person import *
from loss_metric import *


if __name__ == '__main__':
    print('NUM_EPOCHS: ', NUM_EPOCHS)
    print('TIMESTEPS: ', TIMESTEPS)
    print('BATCH_SIZE: ', BATCH_SIZE)
    print('INPUT_SIZE: ', INPUT_SIZE)
    print('INPUT_CHANNELS: ', INPUT_CHANNELS)
    print('NUM_CLASSES: ', NUM_CLASSES)
    print('LEARNING_RATE: ', LEARNING_RATE)

writer = SummaryWriter()

model = UNetDesigner(d1=PARAMETERS['d1'],
                     d2=PARAMETERS['d2'],
                     d3=PARAMETERS['d3'],
                     d4=PARAMETERS['d4'],
                     b_=PARAMETERS['b_'],
                     u4=PARAMETERS['u4'],
                     u3=PARAMETERS['u3'],
                     u2=PARAMETERS['u2'],
                     u1=PARAMETERS['u1']
                     )
model = model.to(device)
model.load_state_dict(torch.load('weights/weights.pth'))

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
'''
Train
'''
iter = 0
val_iter = 0
val_metric = []
print('-'*30)
for epoch in range(NUM_EPOCHS):
    print('*'*10, 'epoch: ', epoch, '*'*10)
    for phase in ['train', 'valid']:
        if phase == 'train':
            loss_list = []
            metric_list = []
            model.train()
            for i, data in enumerate(data_loaders[phase]):
                iter += 1
                input, label, depth = data
                input = input.to(device)
                label = label.to(device)
                depth = depth.to(device)
                output = model(input)
                loss = dice_combo_loss(output, label, depth) ##################### loss
                metric = IoU_metric(output, label)
                loss_list.append(loss.item())
                metric_list.append(metric.item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if iter % 10 == 0:
                    print('loss_iter', iter, ':', loss.item())
                    writer.add_scalar('train_loss', loss.item(), iter)
                    writer.add_scalar('train_metric', metric.item(), iter)
                if iter % 1000 == 0:
                    torch.save(model.state_dict(), 'weights/weights_coco17.pth')
                    print("save_checkpoint")

            train_mean_loss = sum(loss_list) / len(loss_list)
            train_mean_metric = sum(metric_list) / len(metric_list)
            print("train mean_metric: ", train_mean_metric)
        elif phase == 'valid':
            loss_list = []
            metric_list = []
            model.eval()
            for i, data in enumerate(data_loaders[phase]):
                val_iter += 1
                input, label, depth = data
                input = input.to(device)
                label = label.to(device)
                depth = depth.to(device)
                output = model(input)
                loss = dice_combo_loss(output, label, depth) ##################### loss
                metric = IoU_metric(output, label)
                loss_list.append(loss.item())
                metric_list.append(metric.item())

                if val_iter % 50 == 0:
                    print('loss_iter', val_iter, ':', loss.item())
                    writer.add_scalar('train_loss', loss.item(), val_iter)
                    writer.add_scalar('train_metric', metric.item(), val_iter)

            valid_mean_loss = sum(loss_list) / len(loss_list)
            valid_mean_metric = sum(metric_list) / len(metric_list)
            print("valid mean_metric: ", valid_mean_metric)
            val_metric.append(valid_mean_metric)
print('Maximum Valid metric: ', max(val_metric))
print('Tensorboard name: ', GRAPH_NAME)
writer.close()

# !tensorboard --logdir=runs

