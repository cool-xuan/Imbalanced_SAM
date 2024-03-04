import os, random, time, copy
import numpy as np
import os.path as path
import scipy.io as sio
from PIL import Image
from tqdm import tqdm
import sklearn.metrics 

import mindspore as ms
from mindspore import nn
from mindspore import ops

def get_curPerClassAcc(gt, pred):
    confMat = sklearn.metrics.confusion_matrix(gt, pred)                
    # normalize the confusion matrix
    a = confMat.sum(axis=1).reshape((-1,1))
    confMat = confMat / a
    curPerClassAcc = 0
    for i in range(confMat.shape[0]):
        curPerClassAcc += confMat[i,i]
    curPerClassAcc /= confMat.shape[0]
    
    return curPerClassAcc

def forbackward(grad_fn, images, targets, optimizer, optim_mode, tail_classes):
    if optim_mode == 'sgd':
        loss, grads = grad_fn(images, targets)
        loss = loss.mean()
        optimizer(grads)
    elif optim_mode == 'sam':
        loss, grads = grad_fn(images, targets)
        optimizer.first_step(grads)

        loss, grads = grad_fn(images, targets)
        loss = loss.mean()
        optimizer.second_step(grads)
    elif optim_mode == 'imbsam':
        tail_mask = np.where((targets[:, None] == tail_classes[None, :]).sum(1) == 1, True, False)
        head_loss, grads = grad_fn(images, targets, ~tail_mask)
        optimizer.first_step(grads)

        tail_loss, grads = grad_fn(images, targets, tail_mask, True)
        optimizer.second_step(grads)

        tail_loss, grads = grad_fn(images, targets, tail_mask, True)
        optimizer.third_step(grads)
        loss = head_loss.mean() + tail_loss.mean()
    else:
        raise RuntimeError(f'Unknown optimizer mode: {optim_mode}')
    return loss

def train_step(model, dataloader, optimizer, grad_fn, optim_mode, tail_classes):
    model.set_train(True)
    running_loss = 0
    sample_count = 0
    for sample in dataloader.create_dict_iterator():
        images, targets = sample['image'], sample['label']
        targets = targets.view(-1)
        loss = forbackward(grad_fn, images, targets, optimizer, optim_mode, tail_classes)
        sample_count += targets.shape[0]
        running_loss += loss.item() * targets.shape[0]
    
    avg_loss = running_loss / sample_count
    return avg_loss

def eval_step(model, dataloader):
    model.set_train(False)
    predList = np.array([])
    gtList = np.array([])
    for sample in dataloader.create_dict_iterator():
        images, targets = sample['image'], sample['label']
        targets = targets.long().view(-1)
        logits = model(images)

        softmaxScores = ops.softmax(logits, axis=1)
        preds = softmaxScores.argmax(axis=1).float()
        predList = np.concatenate((predList, preds.asnumpy()))
        gtList = np.concatenate((gtList, targets.asnumpy()))

    return get_curPerClassAcc(gtList, predList)

def get_grad_fn(model, optimizer, lossFunc):
    def forward_fn(images, targets, mask=None, is_tail=False):
        logits = model(images)
        if mask is None:
            loss = lossFunc(logits, targets.int())
            loss = loss.mean()
        else:
            logits_selected = ops.masked_select(logits, ms.Tensor(mask[..., None]))
            targets_selected = ops.masked_select(targets, ms.Tensor(mask))
            loss = lossFunc(logits_selected.view(-1, 100), targets_selected.int())
            loss = loss.sum() / targets.shape[0] * 2 if is_tail else loss.sum() / targets.shape[0]
        return loss
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

    return grad_fn

def train_model(dataloaders, model, lossFunc, 
                optimizer,
                num_epochs=50, model_name= 'CE', work_dir='./work_dir', print_each = 1, optim_mode='sgd', tail_classes=None, reweight=1):

    since = time.time()
    best_perClassAcc = 0.0
    epoch = tqdm(range(1, num_epochs+1), total=num_epochs, dynamic_ncols=True)
    grad_fn = get_grad_fn(model, optimizer, lossFunc)

    for i in epoch:  
        loss = train_step(model, dataloaders['train'], optimizer, grad_fn, optim_mode, tail_classes)
        curPerClassAcc = eval_step(model, dataloaders['test'])
        if curPerClassAcc>best_perClassAcc: 
            best_perClassAcc = curPerClassAcc

            path_to_save_param = os.path.join(work_dir, model_name+'_best.ckpt')
            ms.save_checkpoint(model, path_to_save_param)
        
        epoch.set_postfix(loss = loss, acc = curPerClassAcc, best_acc = best_perClassAcc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
