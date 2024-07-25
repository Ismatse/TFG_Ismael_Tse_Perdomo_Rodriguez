import torch
import torch.nn as nn
import math
import copy
from models.resnet import resnet50, resnet18
#from torchvision.models import resnet50, resnet18


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(ProjectionMLP, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True)
        )
        self.l3 = nn.Sequential(
            nn.Linear(mid_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        return x


class PredictionMLP(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(PredictionMLP, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Linear(mid_dim, out_dim)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)

        return x


class SimSiam(nn.Module):

    def __init__(self, backbone='resnet50', d=2048):
        super(SimSiam, self).__init__()

        if backbone == 'resnet50':
            net = resnet50()
        elif backbone == 'resnet18':
            net = resnet18()
        else:
            raise NotImplementedError('Backbone model not implemented.')

        num_ftrs = net.fc.in_features
        self.features = nn.Sequential(*list(net.children())[:-1])
        
        self.feature_maps = nn.Sequential(*list(net.children())[:-2])
        self.aux_list = list(net.children())[:-2]
        # num_ftrs = net.fc.out_features
        # self.features = net

        # projection MLP
        self.projection = ProjectionMLP(num_ftrs, 2048, 2048)
        # prediction MLP
        self.prediction = PredictionMLP(2048, 512, 2048)

        self.reset_parameters()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # projection
        z = self.projection(x)
        # prediction
        p = self.prediction(z)
        return z, p
    
    def get_feature_maps(self, x):
        net_children_for_feature_maps_extraction = copy.deepcopy(self.aux_list)
        for i, layer in enumerate(net_children_for_feature_maps_extraction):
            if isinstance(layer, nn.Sequential):
                for j, sub_layer in enumerate(layer):
                    if j == len(layer) - 1 and i == len(net_children_for_feature_maps_extraction) - 1:
                        sub_layer.conv3 = nn.Identity()
                        sub_layer.bn3 = nn.Identity()
        
        fmaps = nn.Sequential(*net_children_for_feature_maps_extraction)
        x = fmaps(x)
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