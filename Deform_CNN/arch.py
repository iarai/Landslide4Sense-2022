# -*- coding: utf-8 -*-
from model.deform_conv_v2 import *


class DeformCNN(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        features = []
        inplanes = 14
        outplanes = 32

        # The sub-layers consist of two 1-D convolutions and a layer of deformable convolutions in series.
        # args.dcn = number of sub-layers
        for y in range(args.dcn):
            # Adding 2 convolution layers (args.cvn)
            features.append(nn.Conv2d(in_channels=inplanes, out_channels=outplanes,
                                      kernel_size=(1, 3), padding=(0, 1), bias=False))
            features.append(nn.ReLU(inplace=True))
            features.append(nn.MaxPool2d((1, 2)))
            features.append(nn.BatchNorm2d(outplanes))

            for i in range(args.cvn - 1):
                features.append(nn.Conv2d(in_channels=outplanes, out_channels=outplanes,
                                          kernel_size=(1, 3), padding=(0, 1), bias=False))
                features.append(nn.ReLU(inplace=True))
                features.append(nn.MaxPool2d((1, 2)))
                features.append(nn.BatchNorm2d(outplanes))

            # Adding deformable convolution
            features.append(DeformConv2d(inc=outplanes, outc=outplanes, kernel_size=3,
                                         padding=1, bias=False, modulation=args.modulation))
            features.append(nn.ReLU(inplace=True))
            features.append(nn.BatchNorm2d(outplanes))

            if y != args.dcn - 1:
                inplanes = outplanes
                outplanes *= 2

        self.features = nn.Sequential(*features)
        self.fc0 = nn.Linear(in_features=outplanes, out_features=num_classes)

    def forward(self, inputs):
        x = self.features(inputs)
        return x
