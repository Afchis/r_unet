

from args import *
from model_head import *
from dataloader import *
from loss_metric import *


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
for epoch in range(NUM_EPOCHS):# NUM_EPOCHS = 125
    print('*'*10, 'epoch: ', epoch, '*'*10)
    for phase in ['train', 'valid']:
        if phase == 'train':
            loss_list = []
            model.train()
            for i, data in enumerate(data_loaders[phase]):
                input, label, depth = data
                input = input.to(device)
                label = label.to(device)
                depth = depth.to(device)
                output = model(input)
                loss = l2_combo_loss(output, label, depth)
                metric = IoU_metric(output, label)
                loss_list.append(metric.item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            mean_loss = sum(loss_list) / len(loss_list)
            print("train l2_norm: ", mean_loss)
        elif phase == 'valid':
            loss_list = []
            model.eval()
            for i, data in enumerate(data_loaders[phase]):
                input, label, depth = data
                input = input.to(device)
                label = label.to(device)
                depth = depth.to(device)
                output = model(input)
                loss = l2_combo_loss(output, label, depth)
                metric = IoU_metric(output, label)
                loss_list.append(metric.item())
            mean_loss = sum(loss_list) / len(loss_list)
            print("val l2_norm: ", mean_loss)
            val_loss.append(mean_loss)
print('Maximum Valid metric: ', max(val_loss))