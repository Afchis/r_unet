import torch

from args import *
from model_head import *
from dataloader_VOC import *
from loss_metric import *


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
iter = 0
val_metric = []
print('-'*30)
for epoch in range(10):# NUM_EPOCHS = 125
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
                loss = dice_loss(output, label, depth)
                metric = IoU_metric(output, label)
                loss_list.append(loss.item())
                metric_list.append(metric.item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if iter % 1 == 0:
                    print('loss_iter', iter, ':', loss.item())

            train_mean_loss = sum(loss_list) / len(loss_list)
            train_mean_metric = sum(metric_list) / len(metric_list)
            print("train mean_metric: ", train_mean_metric)
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
                loss = dice_loss(output, label, depth)
                metric = IoU_metric(output, label)
                loss_list.append(loss.item())
                metric_list.append(metric.item())

                if iter % 1 == 0:
                    print('loss_iter', iter, ':', loss.item())

            valid_mean_loss = sum(loss_list) / len(loss_list)
            valid_mean_metric = sum(metric_list) / len(metric_list)
            print("valid mean_metric: ", valid_mean_metric)
            val_metric.append(valid_mean_metric)
print('Maximum Valid metric: ', max(val_metric))

# !tensorboard --logdir=runs

#torch.save(model.state_dict(), 'weights/weights.pth')