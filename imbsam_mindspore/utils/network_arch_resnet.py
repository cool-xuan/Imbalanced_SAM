from __future__ import absolute_import, division, print_function
import numpy as np
import os
import mindspore as ms
from mindspore import nn, ops

from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152

class ResnetEncoder(nn.Cell):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers=18, isPretrained=False, isGrayscale=False, embDimension=128, poolSize=4):
        super(ResnetEncoder, self).__init__()
        self.path_to_model = './resnet_ckpts/mindspore'
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.isGrayscale = isGrayscale
        self.isPretrained = isPretrained
        self.embDimension = embDimension
        self.poolSize = poolSize
        self.featListName = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']
        
        resnets = {
            18: resnet18, 
            34: resnet34,
            50: resnet50, 
            101: resnet101,
            152: resnet152}
        
        resnets_pretrained_path = {
            18: 'resnet18-f37072fd.ckpt', 
            34: 'resnet34-b627a593.ckpt',
            50: 'resnet50-0676ba61.ckpt',
            101: 'resnet101-63fe2227.ckpt',
            152: 'resnet152-394f9c45.ckpt'}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(
                num_layers))

        self.encoder = resnets[num_layers]()
        
        if self.isPretrained:
            print("using pretrained model")
            ms.load_checkpoint(os.path.join(self.path_to_model, resnets_pretrained_path[num_layers]), self.encoder)
            # self.encoder.load_state_dict(
            #     torch.load(os.path.join(self.path_to_model, resnets_pretrained_path[num_layers])))
            
        if self.isGrayscale:
            # self.encoder.conv1 = nn.Conv2d(
            #     1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.encoder.conv1 = nn.Conv2d(
                1, 64, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=False)
        else:
            # self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=False)
        
        if num_layers > 34:
            self.num_ch_enc[1:] = 2048
        else:
            self.num_ch_enc[1:] = 512
                    
        if self.embDimension>0:
            self.encoder.fc =  nn.Dense(int(self.num_ch_enc[-1]), int(self.embDimension))
            

    def construct(self, input_image):
        self.features = []
        
        x = self.encoder.conv1(input_image)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        self.features.append(x)
        
        x = self.encoder.layer1(x)
        self.features.append(x)
        
        x = self.encoder.layer2(x)
        self.features.append(x)
        
        x = self.encoder.layer3(x) 
        self.features.append(x)
        
        x = self.encoder.layer4(x)
        self.features.append(x)
        
        # x = F.avg_pool2d(x, self.poolSize)
        x = ops.avg_pool2d(x, self.poolSize, self.poolSize)
        self.x = x.view(x.shape[0], -1)
        
        x = self.encoder.fc(self.x)
        return x
    
    