# -*- coding: utf-8 -*-
# @Time    : 2020-02-26 17:59
# @Author  : Zonas
# @Email   : zonas.wang@gmail.com
# @File    : model.py
"""
"""
import torch.nn as nn
from collections import OrderedDict
import torch
# import torch.nn as nn
import torchvision
# from modules import SuccessiveConv,Decoder_Block,Decoder2_Block,Encoder_Block,ASPP,SELayer

from .unet_base import *
from .nested_unet_base import *
from .densenet_base import *
from .double_unet_base import *


class UNet(nn.Module):
    def __init__(self, img_ch=14, n_classes=2):
        super(UNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(
            64, n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2UNet(nn.Module):
    def __init__(self, img_ch=14, n_classes=2, t=2):
        super(R2UNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(
            64, n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class Att_UNet(nn.Module):
    def __init__(self, img_ch=14, n_classes=2):
        super(Att_UNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(
            64, n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2Att_UNet(nn.Module):
    def __init__(self, img_ch=14, n_classes=2, t=2):
        super(R2Att_UNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(
            64, n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class NestedUNet(nn.Module):
    def __init__(self, n_channels=14, n_classes=2, deepsupervision=False, bilinear=True):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.deepsupervision = deepsupervision
        self.bilinear = bilinear

        nb_filter = [64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(self.n_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(
            nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(
            nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(
            nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(
            nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(
            nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(
            nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(
            nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(
            nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(
            nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(
            nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deepsupervision:
            self.final1 = nn.Conv2d(
                nb_filter[0], self.n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(
                nb_filter[0], self.n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(
                nb_filter[0], self.n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(
                nb_filter[0], self.n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


class DoubleUNet(nn.Module):
    def __init__(self, n_channels=14, n_classes=2):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.enc1_1 = VGGBlock(self.n_channels, 64, 64, True)
        self.enc1_2 = VGGBlock(64, 128, 128, True)
        self.enc1_3 = VGGBlock(128, 256, 256, True)
        self.enc1_4 = VGGBlock(256, 512, 512, True)
        self.enc1_5 = VGGBlock(512, 512, 512, True)

        self.aspp1 = ASPP(512, 512)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dec1_4 = VGGBlock(1024, 256, 256, False)
        self.dec1_3 = VGGBlock(512, 128, 128, False)
        self.dec1_2 = VGGBlock(256, 64, 64, False)
        self.dec1_1 = VGGBlock(128, 32, 32, False)

        self.output1 = output_block()

        self.enc2_1 = VGGBlock(3, 64, 64, True, True)
        self.enc2_2 = VGGBlock(64, 128, 128, True, True)
        self.enc2_3 = VGGBlock(128, 256, 256, True, True)
        self.enc2_4 = VGGBlock(256, 512, 512, True, True)
        self.enc2_5 = VGGBlock(512, 512, 512, True, True)

        self.aspp2 = ASPP(512, 512)

        self.dec2_4 = VGGBlock(1536, 256, 256, False, True)
        self.dec2_3 = VGGBlock(768, 128, 128, False, True)
        self.dec2_2 = VGGBlock(384, 64, 64, False, True)
        self.dec2_1 = VGGBlock(192, 32, 32, False, True)

        self.output2 = output_block(out_channels=self.n_classes)

    def forward(self, _input):
        # encoder of 1st unet
        y_enc1_1 = self.enc1_1(_input)
        y_enc1_2 = self.enc1_2(y_enc1_1)
        y_enc1_3 = self.enc1_3(y_enc1_2)
        y_enc1_4 = self.enc1_4(y_enc1_3)
        y_enc1_5 = self.enc1_5(y_enc1_4)

        # aspp bridge1
        y_aspp1 = self.aspp1(y_enc1_5)

        # decoder of 1st unet
        y_dec1_4 = self.up(y_aspp1)
        y_dec1_4 = self.dec1_4(torch.cat([y_enc1_4, y_dec1_4], 1))
        y_dec1_3 = self.up(y_dec1_4)
        y_dec1_3 = self.dec1_3(torch.cat([y_enc1_3, y_dec1_3], 1))
        y_dec1_2 = self.up(y_dec1_3)
        y_dec1_2 = self.dec1_2(torch.cat([y_enc1_2, y_dec1_2], 1))
        y_dec1_1 = self.up(y_dec1_2)
        y_dec1_1 = self.dec1_1(torch.cat([y_enc1_1, y_dec1_1], 1))
        y_dec1_0 = self.up(y_dec1_1)

        # output of 1st unet
        output1 = self.output1(y_dec1_0)

        # multiply input and output of 1st unet
        mul_output1 = _input * output1

        # encoder of 2nd unet
        y_enc2_1 = self.enc2_1(mul_output1)
        y_enc2_2 = self.enc2_2(y_enc2_1)
        y_enc2_3 = self.enc2_3(y_enc2_2)
        y_enc2_4 = self.enc2_4(y_enc2_3)
        y_enc2_5 = self.enc2_5(y_enc2_4)

        # aspp bridge 2
        y_aspp2 = self.aspp2(y_enc2_5)

        # decoder of 2nd unet
        y_dec2_4 = self.up(y_aspp2)
        y_dec2_4 = self.dec2_4(torch.cat([y_enc1_4, y_enc2_4, y_dec2_4], 1))
        y_dec2_3 = self.up(y_dec2_4)
        y_dec2_3 = self.dec2_3(torch.cat([y_enc1_3, y_enc2_3, y_dec2_3], 1))
        y_dec2_2 = self.up(y_dec2_3)
        y_dec2_2 = self.dec2_2(torch.cat([y_enc1_2, y_enc2_2, y_dec2_2], 1))
        y_dec2_1 = self.up(y_dec2_2)
        y_dec2_1 = self.dec2_1(torch.cat([y_enc1_1, y_enc2_1, y_dec2_1], 1))
        y_dec2_0 = self.up(y_dec2_1)

        # output of 2nd unet
        output2 = self.output2(y_dec2_0)
        final_output = torch.cat((output1, output2))

        return final_output


class DenseNet(nn.Module):
    def __init__(self, kernel_size=(3, 3, 3), n_channels=14, growth_rate=8, block_config=(4, 6, 4),
                 num_init_features=64, bn_size=4, drop_rate=0.25,
                 n_classes=2, memory_efficient=False):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict(
            [('conv0', nn.Conv2d(n_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
             ('norm0', nn.BatchNorm2d(num_init_features)),
             ('relu0', nn.ReLU(inplace=True)),
             ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
             ]))

        # Each denseblock
        num_features = num_init_features
        # i = [0,1,2,3] and num_layers = [6, 12, 24, 16]
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(kernel_size=kernel_size[i], num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                # add transition layer between denseblocks to downsample
                trans = _Transition(
                    num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Classifier layer
        self.classifier = OutConv(num_features, n_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = self.classifier(out)
        return out
