import os, random, time, copy
import numpy as np
import os.path as path
import scipy.io as sio
from scipy import misc
import PIL.Image
import pickle

import mindspore as ms
import mindspore.ops as ops
from mindspore.dataset import GeneratorDataset
from mindspore.dataset import transforms, vision


def get_img_num_per_cls(cls_num, total_num, imb_type, imb_factor):
    # This function is excerpted from a publicly available code [commit 6feb304, MIT License]:
    # https://github.com/kaidic/LDAM-DRW/blob/master/imbalance_cifar.py
    img_max = total_num / cls_num
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)
    return img_num_per_cls

def gen_imbalanced_data(img_num_per_cls, imgList, labelList):
    # This function is excerpted from a publicly available code [commit 6feb304, MIT License]:
    # https://github.com/kaidic/LDAM-DRW/blob/master/imbalance_cifar.py
    new_data = []
    new_targets = []
    targets_np = np.array(labelList, dtype=np.int64)
    classes = np.unique(targets_np)
    # np.random.shuffle(classes)  # remove shuffle in the demo fair comparision
    num_per_cls_dict = dict()
    for the_class, the_img_num in zip(classes, img_num_per_cls):
        num_per_cls_dict[the_class] = the_img_num
        idx = np.where(targets_np == the_class)[0]
        #np.random.shuffle(idx) # remove shuffle in the demo fair comparision
        selec_idx = idx[:the_img_num]
        new_data.append(imgList[selec_idx, ...])
        new_targets.extend([the_class, ] * the_img_num)
    new_data = np.vstack(new_data)
    return (new_data, new_targets)

class CIFAR100LT:
    def __init__(self, set_name='train', imageList=[], labelList=[], labelNames=[], isAugment=True):
        self.isAugment = isAugment
        self.set_name = set_name
        self.labelNames = labelNames
        if self.set_name=='train':            
            self.transform = transforms.Compose([
                vision.RandomCrop(32, padding=4),
                vision.RandomHorizontalFlip(),
                vision.ToTensor(),
                vision.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), is_hwc=False),
            ])
        else:
            self.transform = transforms.Compose([
                vision.ToTensor(),
                vision.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), is_hwc=False),
            ])
        
        self.imageList = imageList
        self.labelList = labelList
        self.current_set_len = len(self.labelList)
        
    def __len__(self):        
        return self.current_set_len
    
    def __getitem__(self, idx):   
        curImage = self.imageList[idx]
        curLabel =  np.asarray(self.labelList[idx])
        # C H W -> H W C
        curImage = PIL.Image.fromarray(curImage.transpose(1,2,0))
        curImage = self.transform(curImage)[0]
        curLabel = np.expand_dims(np.expand_dims(curLabel.astype(np.float32), axis=0), axis=0)

        return curImage, curLabel

def get_loader(dataset, batchsize, shuffle, num_workers=1):
    dataLoader = GeneratorDataset(dataset, column_names=["image", "label"], shuffle=shuffle, num_parallel_workers=num_workers)
    dataLoader = dataLoader.batch(batchsize)
    return dataLoader