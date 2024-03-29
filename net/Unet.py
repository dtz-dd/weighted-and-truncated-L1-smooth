"""
@file: net.py
@intro: this file defines the network structure that is an autoencoder with skip layer.
@date: 2021/03/10 14:31:57
@author: tangling
@version: 1.0
"""


import torch
from torch import nn
import torch.nn.functional as F
# from torchinfo import summary
from torchsummary import summary


class TreeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super(TreeConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv(x)

class FourConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super(FourConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv(x)

class FUnet(nn.Module):
    def __init__(self) -> None:
        super(FUnet, self).__init__()
        self.layer0 = FourConv(5, 11, 3, 1, 1)
        self.layer1 = FourConv(11, 22, 3, 1, 1)
        self.layer2 = FourConv(22, 32, 3, 1, 1)
        self.layer3 = FourConv(32, 64, 3, 1, 1)
        self.layer4 = FourConv(64, 128, 3, 1, 1)
        self.layer5 = FourConv(128, 256, 3, 1, 1)
        self.layer5_1 = FourConv(256, 512, 3, 1, 1)

        self.layer6_1 = FourConv(512, 256, 3, 1, 1)
        self.layer6 = FourConv(256, 128, 3, 1, 1)
        self.layer7 = FourConv(128, 64, 3, 1, 1)
        self.layer8 = FourConv(64, 32, 3, 1, 1)
        self.layer9 = FourConv(32, 22, 3, 1, 1)
        self.layer10 = FourConv(22, 11, 3, 1, 1)
        self.layer11 = FourConv(11, 5, 3, 1, 1)

        self.layer12 = nn.Sequential(
            nn.Conv2d(5, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        d0 = self.layer0(x)
        # d0_h, d0_w = self.getHeightAndWidth(d0)
        d1 = self.layer1(d0)  # d1: [b, 22, 224, 224]
        d1_h, d1_w = self.getHeightAndWidth(d1)
        in1 = F.interpolate(d1, scale_factor=0.5, mode='bilinear')  # in1: [b, 22, 112, 112]
        d2 = self.layer2(in1)  # d2: [b, 32, 112, 112]
        d2_h, d2_w = self.getHeightAndWidth(d2)
        in2 = F.interpolate(d2, scale_factor=0.5, mode='bilinear')  # in2: [b, 32, 56, 56]
        d3 = self.layer3(in2)  # d3: [b, 64, 56, 56]
        d3_h, d3_w = self.getHeightAndWidth(d3)
        in3 = F.interpolate(d3, scale_factor=0.5, mode='bilinear')  # in3: [b, 64, 28, 28]
        d4 = self.layer4(in3)  # d4: [b, 128, 28, 28]
        d4_h, d4_w = self.getHeightAndWidth(d4)
        in4 = F.interpolate(d4, scale_factor=0.5, mode='bilinear')  # in4: [b, 64, 14, 14]
        d5 = self.layer5(in4)  # d5: [b, 256, 14, 14]
        d5_h, d5_w = self.getHeightAndWidth(d5)
        in5 = F.interpolate(d5, scale_factor=0.5, mode='bilinear')  # in5: [b, 256, 7, 7]
        mid = self.layer5_1(in5)  # mid: [b, 512, 7, 7]

        u1_1 = self.layer6_1(mid)  # u1: [b, 256, 7, 7]
        in6_1 = F.interpolate(u1_1, size=(d5_h, d5_w), mode='bilinear')  # in6: [b, 256, 14, 14]
        u1 = self.layer6(in6_1 + d5)  # u1: [b, 128, 14, 14]
        in6 = F.interpolate(u1, size=(d4_h, d4_w), mode='bilinear')  # in6: [b, 128, 28, 28]
        u2 = self.layer7(in6 + d4)  # u2: [b, 64, 28, 28]
        in7 = F.interpolate(u2, size=(d3_h, d3_w), mode='bilinear')  # in7: [b, 64, 56, 56]
        u3 = self.layer8(in7 + d3)  # u3: [b, 32, 56, 56]
        in8 = F.interpolate(u3, size=(d2_h, d2_w), mode='bilinear')  # in8: [b, 32, 112, 112]
        u4 = self.layer9(in8 + d2)  # u4: [b, 10, 112, 112]
        in9 = F.interpolate(u4, size=(d1_h, d1_w), mode='bilinear')  # in9: [b, 10, 224, 224]
        in10 = self.layer10(in9 + d1)
        in11 = self.layer11(in10 + d0)
        out = self.layer12(in11)
        return out

    def getHeightAndWidth(self, x):
        return x.size()[2], x.size()[3]


class Unet(nn.Module):
    def __init__(self) -> None:
        super(Unet, self).__init__()
        self.layer0 = TreeConv(5, 11, 3, 1, 1)
        self.layer1 = TreeConv(11, 22, 3, 1, 1)
        self.layer2 = TreeConv(22, 32, 3, 1, 1)
        self.layer3 = TreeConv(32, 64, 3, 1, 1)
        self.layer4 = TreeConv(64, 128, 3, 1, 1)
        self.layer5 = TreeConv(128, 256, 3, 1, 1)
        self.layer5_1 = TreeConv(256, 512, 3, 1, 1)

        self.layer6_1 = TreeConv(512, 256, 3, 1, 1)
        self.layer6 = TreeConv(256, 128, 3, 1, 1)
        self.layer7 = TreeConv(128, 64, 3, 1, 1)
        self.layer8 = TreeConv(64, 32, 3, 1, 1)
        self.layer9 = TreeConv(32, 22, 3, 1, 1)
        self.layer10 = TreeConv(22, 11, 3, 1, 1)
        self.layer11 = TreeConv(11, 5, 3, 1, 1)

        self.layer12 = nn.Sequential(
            nn.Conv2d(5, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        d0 = self.layer0(x)
        # d0_h, d0_w = self.getHeightAndWidth(d0)
        d1 = self.layer1(d0)  # d1: [b, 22, 224, 224]
        d1_h, d1_w = self.getHeightAndWidth(d1)
        in1 = F.interpolate(d1, scale_factor=0.5, mode='bilinear')  # in1: [b, 22, 112, 112] # 上/下采样
        d2 = self.layer2(in1)  # d2: [b, 32, 112, 112]
        d2_h, d2_w = self.getHeightAndWidth(d2) #改变尺寸
        in2 = F.interpolate(d2, scale_factor=0.5, mode='bilinear')  # in2: [b, 32, 56, 56]
        d3 = self.layer3(in2)  # d3: [b, 64, 56, 56]
        d3_h, d3_w = self.getHeightAndWidth(d3)
        in3 = F.interpolate(d3, scale_factor=0.5, mode='bilinear')  # in3: [b, 64, 28, 28]
        d4 = self.layer4(in3)  # d4: [b, 128, 28, 28]
        d4_h, d4_w = self.getHeightAndWidth(d4)
        in4 = F.interpolate(d4, scale_factor=0.5, mode='bilinear')  # in4: [b, 128, 14, 14]
        d5 = self.layer5(in4)  # d5: [b, 256, 14, 14]
        d5_h, d5_w = self.getHeightAndWidth(d5)
        in5 = F.interpolate(d5, scale_factor=0.5, mode='bilinear')  # in5: [b, 256, 7, 7]
        mid = self.layer5_1(in5)  # mid: [b, 512, 7, 7]

        u1_1 = self.layer6_1(mid)  # u1: [b, 256, 7, 7]
        in6_1 = F.interpolate(u1_1, size=(d5_h, d5_w), mode='bilinear')  # in6: [b, 256, 14, 14]
        u1 = self.layer6(in6_1 + d5)  # u1: [b, 128, 14, 14]
        in6 = F.interpolate(u1, size=(d4_h, d4_w), mode='bilinear')  # in6: [b, 128, 28, 28]
        u2 = self.layer7(in6 + d4)  # u2: [b, 64, 28, 28]
        in7 = F.interpolate(u2, size=(d3_h, d3_w), mode='bilinear')  # in7: [b, 64, 56, 56]
        u3 = self.layer8(in7 + d3)  # u3: [b, 32, 56, 56]
        in8 = F.interpolate(u3, size=(d2_h, d2_w), mode='bilinear')  # in8: [b, 32, 112, 112]
        u4 = self.layer9(in8 + d2)  # u4: [b, 22, 112, 112]
        in9 = F.interpolate(u4, size=(d1_h, d1_w), mode='bilinear')  # in9: [b, 22, 224, 224]
        in10 = self.layer10(in9 + d1)
        in11 = self.layer11(in10 + d0)
        out = self.layer12(in11)
        return out

    def getHeightAndWidth(self, x):
        return x.size()[2], x.size()[3]


class Unet3(nn.Module):
    def __init__(self) -> None:
        super(Unet3, self).__init__()
        self.layer0 = TreeConv(5, 11, 3, 1, 1)
        self.layer1 = TreeConv(11, 22, 3, 1, 1)
        self.layer2 = TreeConv(22, 32, 3, 1, 1)
        self.layer3 = TreeConv(32, 64, 3, 1, 1)
        self.layer4 = TreeConv(64, 128, 3, 1, 1)
        # self.layer5 = TreeConv(128, 256, 3, 1, 1)
        # self.layer5_1 = TreeConv(256, 512, 3, 1, 1)

        # self.layer6_1 = TreeConv(512, 256, 3, 1, 1)
        # self.layer6 = TreeConv(256, 128, 3, 1, 1)
        self.layer7 = TreeConv(128, 64, 3, 1, 1)
        self.layer8 = TreeConv(64, 32, 3, 1, 1)
        self.layer9 = TreeConv(32, 22, 3, 1, 1)
        self.layer10 = TreeConv(22, 11, 3, 1, 1)
        self.layer11 = TreeConv(11, 5, 3, 1, 1)

        self.layer12 = nn.Sequential(
            nn.Conv2d(5, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        d0 = self.layer0(x)
        # d0_h, d0_w = self.getHeightAndWidth(d0)
        d1 = self.layer1(d0)  # d1: [b, 22, 224, 224]
        d1_h, d1_w = self.getHeightAndWidth(d1)
        in1 = F.interpolate(d1, scale_factor=0.5, mode='bilinear')  # in1: [b, 22, 112, 112]
        d2 = self.layer2(in1)  # d2: [b, 32, 112, 112]
        d2_h, d2_w = self.getHeightAndWidth(d2)
        in2 = F.interpolate(d2, scale_factor=0.5, mode='bilinear')  # in2: [b, 32, 56, 56]
        d3 = self.layer3(in2)  # d3: [b, 64, 56, 56]
        d3_h, d3_w = self.getHeightAndWidth(d3)
        in3 = F.interpolate(d3, scale_factor=0.5, mode='bilinear')  # in3: [b, 64, 28, 28]
        d4 = self.layer4(in3)  # d4: [b, 128, 28, 28]
        # d4_h, d4_w = self.getHeightAndWidth(d4)
        # in4 = F.interpolate(d4, scale_factor=0.5, mode='bilinear')  # in4: [b, 64, 14, 14]
        # d5 = self.layer5(in4)  # d5: [b, 256, 14, 14]
        # d5_h, d5_w = self.getHeightAndWidth(d5)
        # in5 = F.interpolate(d5, scale_factor=0.5, mode='bilinear')  # in5: [b, 256, 7, 7]
        # mid = self.layer5_1(in5)  # mid: [b, 512, 7, 7]

        # u1_1 = self.layer6_1(mid)  # u1: [b, 256, 7, 7]
        # in6_1 = F.interpolate(u1_1, size=(d5_h, d5_w), mode='bilinear')  # in6: [b, 256, 14, 14]
        # u1 = self.layer6(in6_1 + d5)  # u1: [b, 128, 14, 14]
        # in6 = F.interpolate(u1, size=(d4_h, d4_w), mode='bilinear')  # in6: [b, 128, 28, 28]
        u2 = self.layer7(d4)  # u2: [b, 64, 28, 28]
        in7 = F.interpolate(u2, size=(d3_h, d3_w), mode='bilinear')  # in7: [b, 64, 56, 56]
        u3 = self.layer8(in7 + d3)  # u3: [b, 32, 56, 56]
        in8 = F.interpolate(u3, size=(d2_h, d2_w), mode='bilinear')  # in8: [b, 32, 112, 112]
        u4 = self.layer9(in8 + d2)  # u4: [b, 10, 112, 112]
        in9 = F.interpolate(u4, size=(d1_h, d1_w), mode='bilinear')  # in9: [b, 10, 224, 224]
        in10 = self.layer10(in9 + d1)
        in11 = self.layer11(in10 + d0)
        out = self.layer12(in11)
        return out

    def getHeightAndWidth(self, x):
        return x.size()[2], x.size()[3]

if __name__ == "__main__":
    model = Unet()
    # x = torch.rand((3, 5, 221, 221), requires_grad=True)
    model = model.to(torch.device("cuda"))
    # x = x.to(torch.device("cuda"))
    x = torch.randn((5, 224, 224))
    x = x.to(torch.device("cuda"))
    summary(model, (5, 224, 224))
    summary(model, (x))
    # summary(model, x)
    # y = model(x)
    # print(x.shape)
    # print(y.shape)
    
