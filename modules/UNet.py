# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import unetConv2, unetUp, unetUpCat, unetGateAttention
from init_weights import init_weights


class UNet(nn.Module):
    def __init__(self, in_channels=14, n_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        # filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUpCat(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUpCat(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUpCat(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUpCat(filters[1], filters[0], self.is_deconv)

        self.outconv1 = nn.Conv2d(filters[0], n_classes, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)              # 16*512*1024
        maxpool1 = self.maxpool1(conv1)         # 16*256*512

        conv2 = self.conv2(maxpool1)            # 32*256*512
        maxpool2 = self.maxpool2(conv2)         # 32*128*256

        conv3 = self.conv3(maxpool2)            # 64*128*256
        maxpool3 = self.maxpool3(conv3)         # 64*64*128

        conv4 = self.conv4(maxpool3)            # 128*64*128
        maxpool4 = self.maxpool4(conv4)         # 128*32*64

        center = self.center(maxpool4)          # 256*32*64

        up4 = self.up_concat4(center, conv4)    # 128*64*128
        up3 = self.up_concat3(up4, conv3)       # 64*128*256
        up2 = self.up_concat2(up3, conv2)       # 32*256*512
        up1 = self.up_concat1(up2, conv1)       # 16*512*1024

        d1 = self.outconv1(up1)                 # 256

        return F.sigmoid(d1)


class UNet_Att(nn.Module):
    def __init__(self, in_channels=14, n_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet_Att, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        # filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        self.up5 = unetUp(filters[4], filters[3])
        self.att5 = unetGateAttention(filters[3], filters[3], filters[2])
        self.up_conv5 = unetUpCat(filters[4], filters[3], self.is_deconv)

        self.up4 = unetUp(filters[3], filters[2])
        self.att4 = unetGateAttention(filters[2], filters[2], filters[1])
        self.up_conv4 = unetUpCat(filters[3], filters[2])

        self.up3 = unetUp(filters[2], filters[1])
        self.att3 = unetGateAttention(filters[1], filters[1], filters[0])
        self.up_conv3 = unetUpCat(filters[2], filters[1])

        self.up2 = unetUp(filters[1], filters[0])
        self.att2 = unetGateAttention(filters[0], filters[0], 32)
        self.up_conv2 = unetUpCat(filters[1], filters[0])

        self.conv_1x1 = nn.Conv2d(64, n_classes, 1, 1, 0)

    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)

        x2 = self.maxpool1(x1)
        x2 = self.conv2(x2)

        x3 = self.maxpool2(x2)
        x3 = self.conv3(x3)

        x4 = self.maxpool3(x3)
        x4 = self.conv4(x4)

        x5 = self.maxpool4(x4)
        x5 = self.conv5(x5)

        # decoding + concat path
        d5 = self.up5(x5)
        x4 = self.att5(d5, x4)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        x3 = self.att4(d4, x3)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        x2 = self.att3(d3, x2)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        x1 = self.att2(d2, x=x1)
        d2 = self.up_conv2(d2)

        d1 = self.conv_1x1(d2)

        return d1
