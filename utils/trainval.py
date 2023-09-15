import os, random, time, copy
from skimage import io, transform
import numpy as np
import os.path as path
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
import sklearn.metrics 
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import models, transforms



def train_model(dataloaders, model, lossFunc, 
                optimizer, scheduler,
                num_epochs=50, model_name= 'CE', work_dir='./work_dir', device='cuda', print_each = 1, optim_mode='sgd', tail_classes=None, reweight=1):

    since = time.time()
    best_perClassAcc = 0.0
    
    phases = ['train', 'test']
    
    
    for epoch in range(num_epochs):  
        if epoch%print_each==0:
            print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            if epoch%print_each==0:
                print(phase)
            
            predList = np.array([])
            grndList = np.array([])
            
            if phase == 'train':
                scheduler.step()                
                model.train()
            else:
                model.eval()  # Set model to training mode  
              
            running_loss = 0.0
            running_acc = 0.0
            
            # Iterate over data.
            iterCount, sampleCount = 0, 0
            for sample in dataloaders[phase]:                
                images, targets = sample
                images = images.to(device)
                targets = targets.type(torch.long).view(-1).to(device)

                with torch.set_grad_enabled(phase=='train'):
                    logits = model(images)
                    loss = lossFunc(logits, targets)
                    softmaxScores = logits.softmax(dim=1)

                    preds = softmaxScores.argmax(dim=1).detach().squeeze().type(torch.float)                  
                    accRate = (targets.type(torch.float).squeeze() - preds.squeeze().type(torch.float))
                    accRate = (accRate==0).type(torch.float).mean()
                    
                    predList = np.concatenate((predList, preds.cpu().numpy()))
                    grndList = np.concatenate((grndList, targets.cpu().numpy()))
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss = forbackward(model, lossFunc, optimizer, optim_mode, tail_classes, images, targets, loss)
                    else:
                        loss = loss.mean()
                        
                # statistics  
                iterCount += 1
                sampleCount += targets.size(0)
                running_acc += accRate*targets.size(0) 
                running_loss += loss.item() * targets.size(0) 
                
                print2screen_avgLoss = running_loss / sampleCount
                print2screen_avgAccRate = running_acc / sampleCount
                
            epoch_error = print2screen_avgLoss      
            
            confMat = sklearn.metrics.confusion_matrix(grndList, predList)                
            # normalize the confusion matrix
            a = confMat.sum(axis=1).reshape((-1,1))
            confMat = confMat / a
            curPerClassAcc = 0
            for i in range(confMat.shape[0]):
                curPerClassAcc += confMat[i,i]
            curPerClassAcc /= confMat.shape[0]
            if epoch%print_each==0:
                print('\tloss:{:.6f}, acc-all:{:.5f}, acc-avg-cls:{:.5f}'.format(
                    epoch_error, print2screen_avgAccRate, curPerClassAcc))

            if (phase=='val' or phase=='test') and curPerClassAcc>best_perClassAcc: 
                best_perClassAcc = curPerClassAcc

                path_to_save_param = os.path.join(work_dir, model_name+'_best.pth')
                torch.save(model.state_dict(), path_to_save_param)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    

def forbackward(model, lossFunc, optimizer, optim_mode, tail_classes, images, targets, loss):
    if optim_mode == 'sgd':
        optimizer.zero_grad()
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        
    elif optim_mode == 'sam':
        loss = loss.mean()
        loss.backward()
        optimizer.first_step()
        
        logits = model(images)
        loss = lossFunc(logits, targets)
        loss = loss.mean()
        loss.backward()
        optimizer.second_step()
        
    elif optim_mode == 'imbsam':
        tail_mask = torch.where((targets[:, None] == tail_classes[None, :].to(targets.device)).sum(1) == 1, True, False)
                            
        head_loss = loss[~tail_mask].sum() / targets.size(0) 
        head_loss.backward(retain_graph=True)
        optimizer.first_step()
                            
        tail_loss = loss[tail_mask].sum() / targets.size(0) * 2
        tail_loss.backward()
        optimizer.second_step()
                            
        logits = model(images)
        tail_loss = lossFunc(logits[tail_mask], targets[tail_mask]).sum() / targets.size(0) * 2
        tail_loss.backward()
        optimizer.third_step()
        loss = head_loss + tail_loss
        
    return loss
