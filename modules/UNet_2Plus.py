# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models

from modules.layers import unetConv2, unetUp_origin
from modules.init_weights import init_weights
from modules.layers import ChannelAttention, Attention_block


# L4 (https://pub.towardsai.net/unet-clearly-explained-a-better-image-segmentation-architecture-f48661c92df9)
class UNet_2Plus(nn.Module):
    def __init__(self, in_channels=14, n_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True, is_ds=True):
        super(UNet_2Plus, self).__init__()

        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv00 = unetConv2(
            self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)

        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat01 = unetUp_origin(
            filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp_origin(
            filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp_origin(
            filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp_origin(
            filters[4], filters[3], self.is_deconv)

        self.up_concat02 = unetUp_origin(
            filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp_origin(
            filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp_origin(
            filters[3], filters[2], self.is_deconv, 3)

        self.up_concat03 = unetUp_origin(
            filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = unetUp_origin(
            filters[2], filters[1], self.is_deconv, 4)

        self.up_concat04 = unetUp_origin(
            filters[1], filters[0], self.is_deconv, 5)

        # final conv (without any concat)
        # self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        # self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        # self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        # self.final_4 = nn.Conv2d(filters[0], n_classes, 1)

        self.ca = ChannelAttention(filters[0] * 4, 16)
        self.ca1 = ChannelAttention(filters[0], 16 // 4)

        self.conv_final = nn.Conv2d(filters[0] * 4, n_classes, kernel_size=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)
        maxpool0 = self.maxpool0(X_00)

        X_10 = self.conv10(maxpool0)
        maxpool1 = self.maxpool1(X_10)

        X_20 = self.conv20(maxpool1)
        maxpool2 = self.maxpool2(X_20)

        X_30 = self.conv30(maxpool2)
        maxpool3 = self.maxpool3(X_30)

        X_40 = self.conv40(maxpool3)

        # column : 1
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40, X_30)

        # column : 2
        X_02 = self.up_concat02(X_11, X_00, X_01)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        X_22 = self.up_concat22(X_31, X_20, X_21)

        # column : 3
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)

        # column : 4
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)

        # final layer
        # final_1 = self.final_1(X_01)
        # final_2 = self.final_2(X_02)
        # final_3 = self.final_3(X_03)
        # final_4 = self.final_4(X_04)

        # final = (final_1 + final_2 + final_3 + final_4) / 4

        # if self.is_ds:
        #     return final
        # else:
        #     return final_4

        out = torch.cat([X_01, X_02, X_03, X_04], 1)

        intra = torch.sum(torch.stack((X_01, X_02, X_03, X_04)), dim=0)
        ca1 = self.ca1(intra)
        out = self.ca(out) * (out + ca1.repeat(1, 4, 1, 1))
        out = self.conv_final(out)

        return out


