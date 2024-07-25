# Reference: CutMix repo
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=2, stride=2
    )


class EndToEnd(nn.Module):
    def __init__(self, contrastive, generative, num_classes):
        super(EndToEnd, self).__init__()

        self.contrastive = contrastive
        self.encoder_layers = list(self.contrastive.children())

        self.block1 = nn.Sequential(*self.encoder_layers[0][:3])
        self.maxpool = self.encoder_layers[0][3]
        self.block2 = nn.Sequential(*self.encoder_layers[0][4])
        self.block3 = nn.Sequential(*self.encoder_layers[0][5])
        self.block4 = nn.Sequential(*self.encoder_layers[0][6])
        self.block5 = nn.Sequential(*self.encoder_layers[0][7])

        self.generative_layers = list(generative.children())
        self.generative = generative
        self.to_feature_maps = Rearrange("b (h w) c -> b c h w", h=16, w=32)

        self.up_conv6 = up_conv(512, 512)
        self.conv6 = double_conv(512 + 1024, 512)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = double_conv(256 + 512, 256)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = double_conv(128 + 256, 128)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = double_conv(64 + 64, 64)
        self.up_conv10 = up_conv(64, 32)
        self.conv10 = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        block1 = self.block1(x)
        maxpool = self.maxpool(block1)
        block2 = self.block2(maxpool)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        x = self.generative(block5)
        x = self.to_feature_maps(x)

        x = self.up_conv6(x)
        x = torch.cat([x, block4], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x, block3], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, block2], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x = torch.cat([x, block1], dim=1)
        x = self.conv9(x)

        x = self.up_conv10(x)
        x = self.conv10(x)

        return x

