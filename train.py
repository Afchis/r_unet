from torch.utils.tensorboard import SummaryWriter

from args import *
from model_head import *
from dataloader import *
from loss_metric import *


writer = SummaryWriter()

model = UNetDesigner(d1=PARAMETERS['d1'],
                     d2=PARAMETERS['d2'],
                     d3=PARAMETERS['d3'],
                     b_=PARAMETERS['b_'],
                     u1=PARAMETERS['u1'],
                     u2=PARAMETERS['u2'],
                     u3=PARAMETERS['u3'],
                     cell_model=PARAMETERS['cell_model']
                     )
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

'''
Train
'''
val_loss = []
for epoch in range(5):# NUM_EPOCHS = 125
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
                loss = dice_combo_loss(output, label, depth)
                metric = IoU_metric(output, label)
                loss_list.append(loss.item())
                metric_list.append(metric.item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            mean_loss = sum(loss_list) / len(loss_list)
            mean_metric = sum(metric_list) / len(metric_list)
            writer.add_scalar('loss/train', mean_loss, epoch)
            writer.add_scalar('metric/train', mean_metric, epoch)
            print("train l2_norm: ", mean_metric)
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
                loss = dice_combo_loss(output, label, depth)
                metric = IoU_metric(output, label)
                loss_list.append(loss.item())
                metric_list.append(metric.item())
            mean_loss = sum(loss_list) / len(loss_list)
            mean_metric = sum(metric_list) / len(metric_list)
            writer.add_scalar('loss/valid', mean_loss, epoch)
            writer.add_scalar('metric/valid', mean_metric, epoch)
            print("val l2_norm: ", mean_metric)
            val_loss.append(mean_metric)
print('Maximum Valid metric: ', max(val_loss))
writer.close()
# !tensorboard --logdir=runs

#torch.save(model.state_dict(), 'weights/weights.pth')