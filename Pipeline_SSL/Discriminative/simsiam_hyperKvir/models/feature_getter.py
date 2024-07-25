import torch
import torch.nn as nn
import math
import copy
from models.modified_resnet import resnet50, resnet18
#from torchvision.models import resnet50, resnet18


class FeatureGetter(nn.Module):

    def __init__(self, backbone='resnet50', d=2048):
        super(FeatureGetter, self).__init__()

        if backbone == 'resnet50':
            net = resnet50()
        elif backbone == 'resnet18':
            net = resnet18()
        else:
            raise NotImplementedError('Backbone model not implemented.')
        
        self.features = nn.Sequential(*list(net.children()))

        self.reset_parameters()

    def forward(self, x):
        x = self.features(x)

        return x
    
    def reset_parameters(self):
        # reset conv initialization to default uniform initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                stdv = 1. / math.sqrt(n)
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)