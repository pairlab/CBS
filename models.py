import torch
import torch.nn as nn
import torch.nn.functional as F

import math

def get_gaussian_filter(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    if kernel_size == 3:
        padding = 1
    else:
        padding = 0
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels,
                                bias=False, padding=padding)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter


class CNNNormal(nn.Module):
    def __init__(self, nc, num_classes, std=1):
        super(CNNNormal, self).__init__()

        self.conv1 = nn.Conv2d(nc, 32, kernel_size=3, padding=1)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(256*2*2, 256*2*2)
        self.classifier = nn.Linear(256*2*2, num_classes)

        self.std = std


    def get_new_kernels(self, epoch_count, total_epochs):
        if epoch_count % 10 == 0:
            self.std *= 0.925
        self.kernel0 = get_gaussian_filter(kernel_size=3, sigma=self.std/1, channels=3)
        self.kernel1 = get_gaussian_filter(kernel_size=3, sigma=self.std/1, channels=32)
        self.kernel2= get_gaussian_filter(kernel_size=3, sigma=self.std/1, channels=64)
        self.kernel3 = get_gaussian_filter(kernel_size=3, sigma=self.std/1, channels=128)
        self.kernel4 = get_gaussian_filter(kernel_size=3, sigma=self.std/1, channels=256)


    def forward(self, x):
        x = self.conv1(x)
        x = self.kernel1(x)
        x = F.relu(self.max1(x))

        x = self.conv2(x)
        x = self.kernel2(x)
        x = F.relu(self.max2(x))

        x = self.conv3(x)
        x = self.kernel3(x)
        x = F.relu(self.max3(x))

        x = self.conv4(x)
        x = self.kernel4(x)

        x = F.relu(self.max4(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc(x))
         
        x = self.classifier(x)

        return x


class SimpleMLP(nn.Module):
    def __init__(self, num_classes, input_dim):
        super(SimpleMLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class OneLayerMLP(nn.Module):
    def __init__(self, num_classes, input_dim):
        super(OneLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        return x
