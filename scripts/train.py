from __future__ import print_function, division
import os, pickle, sys
import numpy as np
import os.path as path
import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torchvision

import sys
sys.path.append('../')
from utils.eval_funcs import print_accuracy
from utils.dataset_CIFAR100LT import CIFAR100LT, get_img_num_per_cls, gen_imbalanced_data
from utils.network_arch_resnet import ResnetEncoder
from utils.trainval import train_model
import warnings # ignore warnings
warnings.filterwarnings("ignore")
print(sys.version)
print(torch.__version__)
from imbsam import SAM, ImbSAM

def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=9826)
    parser.add_argument('--print_freq', type=int, default=10)
    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('-bz', '--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--imb_factor', type=float, default=0.01)
    # optimizer
    parser.add_argument('--opt', type=str, default='sgd', choices=['sgd', 'sam', 'imbsam'])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.001)
    parser.add_argument('--rho', type=float, default=0.05)
    parser.add_argument('--eta', type=int, default=20)
    parser.add_argument('--reweight', type=float, default=1.)

    args = parser.parse_args()
    
    return args

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # set device, which gpu to use.
    device ='cpu'
    if torch.cuda.is_available(): 
        device='cuda'
        torch.cuda.device_count()
        torch.cuda.empty_cache()
        
    # number of classes in CIFAR100-LT with imbalance factor 100
    no_of_classes = 100

    dataloaders, train_samples_per_cls, train_few_classes, train_labelList = prepare_data(args, no_of_classes)
        
    print('{} train dataset have {} samples for the head class and {} samples for tail class.'.format(args.dataset, max(train_samples_per_cls), min(train_samples_per_cls)))
    
    # get the execute path
    curr_working_dir = os.getcwd()
    
    model_name = 'CE'

    opt_name = args.opt
    if 'imb' in opt_name:
        opt_name = os.path.join(opt_name, 'eta{}'.format(args.eta))
    if 'sam' in opt_name:
        opt_name = os.path.join(opt_name, 'rho{}'.format(args.rho))
        
    save_dir = path.join(curr_working_dir, 'work_dir', args.dataset,
                         model_name,
                         opt_name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    model = ResnetEncoder(34, False, embDimension=no_of_classes, poolSize=4).to(device)

    
    train(args, device, model_name, model, save_dir, dataloaders, train_few_classes)
    test(device, model_name, save_dir, dataloaders, train_labelList, model)

def test(device, model_name, save_dir, dataloaders, train_labelList, model):
    path_to_clsnet = os.path.join(save_dir, '{}_best.pth'.format(model_name))
    model.load_state_dict(torch.load(path_to_clsnet, map_location=device))
    model = model.to(device)

    print('Testing....'.format(model_name))
    print_accuracy(model, dataloaders, train_labelList, device = device, save_dir=save_dir)


def train(args, device, model_name, model, save_dir, dataloaders, train_few_classes):
    
    loss_func = nn.CrossEntropyLoss(reduction='none').to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    
    if args.opt == 'sam':
        optimizer = SAM(optimizer=optimizer, model=model, rho=args.rho)
    elif args.opt == 'imbsam':
        optimizer = ImbSAM(optimizer=optimizer, model=model, rho=args.rho)

    train_model(dataloaders, model, loss_func, optimizer, scheduler, 
                num_epochs=args.epochs, model_name= model_name, work_dir=save_dir, 
                device=device, print_freq=args.print_freq,
                optim_mode=args.opt, few_classes=train_few_classes, reweight=args.reweight)

    return model


def prepare_data(args, no_of_classes):
    path_to_DB = './datasets'
    if not os.path.exists(path_to_DB): os.makedirs(path_to_DB)
    _ = torchvision.datasets.CIFAR100(root=path_to_DB, train=True, download=True)

    path_to_DB = path.join(path_to_DB, 'cifar-100-python')

    datasets = {}
    dataloaders = {}

    setname = 'meta'
    with open(os.path.join(path_to_DB, setname), 'rb') as obj:
        labelnames = pickle.load(obj, encoding='bytes')
        labelnames = labelnames[b'fine_label_names']
    for i in range(len(labelnames)):
        labelnames[i] = labelnames[i].decode("utf-8") 
        
        
    setname = 'train'
    with open(os.path.join(path_to_DB, setname), 'rb') as obj:
        DATA = pickle.load(obj, encoding='bytes')
    imgList = DATA[b'data'].reshape((DATA[b'data'].shape[0],3, 32,32))
    labelList = DATA[b'fine_labels']
    total_num = len(labelList)
    train_samples_per_cls = get_img_num_per_cls(no_of_classes, total_num, 'exp', args.imb_factor)
    train_few_classes = torch.where(torch.tensor(train_samples_per_cls)<args.eta)[0]
    train_imgList, train_labelList = gen_imbalanced_data(train_samples_per_cls, imgList, labelList)
    datasets[setname] = CIFAR100LT(
        imageList=train_imgList, labelList=train_labelList, labelNames=labelnames,
        set_name=setname, isAugment=setname=='train')
    print('#examples in {}-set:'.format(setname), datasets[setname].current_set_len)

    setname = 'test'
    with open(os.path.join(path_to_DB, setname), 'rb') as obj:
        DATA = pickle.load(obj, encoding='bytes')
    imgList = DATA[b'data'].reshape((DATA[b'data'].shape[0],3, 32,32))
    labelList = DATA[b'fine_labels']
    total_num = len(labelList)
    datasets[setname] = CIFAR100LT(
        imageList=imgList, labelList=labelList, labelNames=labelnames,
        set_name=setname, isAugment=setname=='train')
    print('#examples in {}-set:'.format(setname), datasets[setname].current_set_len)

    dataloaders = {set_name: DataLoader(datasets[set_name],
                                        batch_size=args.batch_size,
                                        shuffle=set_name=='train', 
                                        num_workers=os.cpu_count() // 4) # num_work can be set to batch_size
                   for set_name in ['train', 'test']}

    print('#train batch:', len(dataloaders['train']), '\t#test batch:', len(dataloaders['test']))
    return dataloaders, train_samples_per_cls, train_few_classes, train_labelList

if __name__ == '__main__':
    args = get_parser()
    main(args)