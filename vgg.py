import torch
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

from utils import *


class VGG16_conv(torch.nn.Module):
    def __init__(self, n_classes, args):
        super(VGG16_conv, self).__init__()
        self.std = args.std
        self.factor = args.std_factor
        self.epoch = args.epoch
        self.kernel_size = args.kernel_size

        self.conv1 = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, 3, padding=1),
        )
        self.post1 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2, stride=2)
        )
        self.conv2 = torch.nn.Sequential(
                torch.nn.Conv2d(64, 128, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 128, 3, padding=1),
        )
        self.post2 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2, stride=2)
        )
        self.conv3 = torch.nn.Sequential(
                torch.nn.Conv2d(128, 256, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 256, 3, padding=1),
        )
        self.post3 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2, stride=2)
        )
        self.conv4 = torch.nn.Sequential(
                torch.nn.Conv2d(256, 512, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(512, 512, 3, padding=1),
        )
        self.post4 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2, stride=2)
        )
        self.conv5 = torch.nn.Sequential(
                torch.nn.Conv2d(512, 512, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(512, 512, 3, padding=1),
        )
        self.post5 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2, stride=2)
        )

        self.classifier = torch.nn.Sequential(
                torch.nn.Linear(512, 4096),
                torch.nn.ReLU(),
                torch.nn.Dropout(),
                torch.nn.Linear(4096, 4096),
                torch.nn.ReLU(),
                torch.nn.Dropout(),
                torch.nn.Linear(4096, n_classes)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def get_new_kernels(self, epoch_count):
        if epoch_count % self.epoch == 0 and epoch_count is not 0:
            self.std *= 0.9

        self.kernel1 = get_gaussian_filter(
                kernel_size=self.kernel_size,
                sigma=self.std,
                channels=64
        )

        self.kernel2= get_gaussian_filter(
                kernel_size=self.kernel_size,
                sigma=self.std,
                channels=128
        )

        self.kernel3 = get_gaussian_filter(
                kernel_size=self.kernel_size,
                sigma=self.std,
                channels=256
        )

        self.kernel4 = get_gaussian_filter(
                kernel_size=self.kernel_size,
                sigma=self.std,
                channels=512
        )

        self.kernel5 = get_gaussian_filter(
                kernel_size=self.kernel_size,
                sigma=self.std,
                channels=512
        )


    def forward(self, x, return_intermediate=False):
        output = self.conv1(x)
        output = self.kernel1(output) 
        output = self.post1(output)

        output = self.conv2(output)
        output = self.kernel2(output) 
        output = self.post2(output)

        output = self.conv3(output)
        output = self.kernel3(output) 
        output = self.post3(output)

        output = self.conv4(output)
        output = self.kernel4(output) 
        output = self.post4(output)

        output = self.conv5(output)
        output = self.kernel5(output) 

        if return_intermediate:
            output = output.view(output.size(0), -1)
            return output

        output = self.post5(output)

        output = output.view(output.size(0), -1)
        output = self.classifier(output)

        return output
