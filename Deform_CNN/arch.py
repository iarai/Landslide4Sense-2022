# -*- coding: utf-8 -*-
from model.deform_conv_v2 import *
import Deform_CNN.smu as smu


class DeformCNN(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        self.smu = smu.SMU()
        # self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

        features = []
        inplanes = 14
        outplanes = 64

        # The sub-layers consist of two 1-D convolutions and a layer of deformable convolutions in series.
        # args.dcn = number of sub-layers
        for y in range(args.dcn):
            # Adding 2 convolution layers (args.cvn)
            features.append(nn.Conv2d(in_channels=inplanes, out_channels=outplanes,
                                      kernel_size=3, padding=1, bias=False))
            features.append(self.smu)
            features.append(self.pool)
            features.append(nn.BatchNorm2d(outplanes))

            for i in range(args.cvn - 1):
                features.append(nn.Conv2d(in_channels=outplanes, out_channels=outplanes,
                                          kernel_size=3, padding=1, bias=False))
                features.append(self.smu)
                features.append(self.pool)
                features.append(nn.BatchNorm2d(outplanes))

            # Adding deformable convolution
            features.append(DeformConv2d(inc=outplanes, outc=outplanes, kernel_size=3,
                                         padding=1, bias=False, modulation=args.modulation))
            features.append(self.smu)
            features.append(nn.BatchNorm2d(outplanes))

            if y != args.dcn - 1:
                inplanes = outplanes
                outplanes *= 2

        self.features = nn.Sequential(*features)

        self.fc1 = nn.Conv2d(in_channels=outplanes, out_channels=outplanes, kernel_size=1)
        self.fc0 = nn.Conv2d(in_channels=outplanes, out_channels=num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self.features(inputs)
        # x = self.fc1(x)
        # x = self.smu(x)
        output = self.fc0(x)
        return output
