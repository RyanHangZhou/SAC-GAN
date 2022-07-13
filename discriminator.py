import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
# import utils

from torchvision.utils import save_image
from collections import namedtuple

import ResNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sem_len = 128
obj_len = 30
nClass = 19 # 30



class TDisNet(nn.Module):
    def __init__(self):
        super(TDisNet, self).__init__()

        f_dim = 32

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, f_dim, kernel_size=1),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(f_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e0 = self.conv0(x.reshape([-1, 6, 1, 1]))
        disc = self.conv1(e0)

        return disc

class SemDisNet(nn.Module):
    def __init__(self):
        super(SemDisNet, self).__init__()

        f_dim = 8

        self.conv0 = nn.Sequential(
            nn.Conv2d((nClass + 1), f_dim * 1,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(f_dim * 1, f_dim * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 2, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(f_dim * 2, f_dim * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 4, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(f_dim * 4, 1,
                      kernel_size=4, stride=2, padding=1),
            # nn.Sigmoid()
            nn.InstanceNorm2d(1, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(1, 1,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(1, 1,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(1, 1,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(1, 2,
                      kernel_size=(5, 8), stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, seg, mask):
        d0 = self.conv0(torch.cat([seg, mask], 1))
        d1 = self.conv1(d0)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        d5 = self.conv5(d4)
        d6 = self.conv6(d5)
        d7 = self.conv7(d6)
        disc = d7.view([1, 2])
        return disc

# class SemDisNet(nn.Module):
#     def __init__(self):
#         super(SemDisNet, self).__init__()

#         f_dim = 8

#         self.conv0 = nn.Sequential(
#             nn.Conv2d(nClass, f_dim * 1,
#                       kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#         )
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(f_dim * 1, f_dim * 2,
#                       kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(f_dim * 2, affine=False),
#             nn.LeakyReLU(0.2),
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(f_dim * 2, f_dim * 4,
#                       kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(f_dim * 4, affine=False),
#             nn.LeakyReLU(0.2),
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(f_dim * 4, 1,
#                       kernel_size=4, stride=2, padding=1),
#             # nn.Sigmoid()
#             nn.InstanceNorm2d(1, affine=False),
#             nn.LeakyReLU(0.2),
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(1, 1,
#                       kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(1, affine=False),
#             nn.LeakyReLU(0.2),
#         )
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(1, 1,
#                       kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(1, affine=False),
#             nn.LeakyReLU(0.2),
#         )
#         self.conv6 = nn.Sequential(
#             nn.Conv2d(1, 1,
#                       kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(1, affine=False),
#             nn.LeakyReLU(0.2),
#         )
#         self.conv7 = nn.Sequential(
#             nn.Conv2d(1, 2,
#                       kernel_size=(5, 8), stride=2, padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, seg):
#         d0 = self.conv0(seg)
#         d1 = self.conv1(d0)
#         d2 = self.conv2(d1)
#         d3 = self.conv3(d2)
#         d4 = self.conv4(d3)
#         d5 = self.conv5(d4)
#         d6 = self.conv6(d5)
#         d7 = self.conv7(d6)
#         disc = d7.view([1, 2])
#         return disc


class Composability(nn.Module):
    def __init__(self):
        super(Composability, self).__init__()

        f_dim = 8

        self.conv0 = nn.Sequential(
            nn.Conv2d(nClass, f_dim * 1,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(f_dim * 1, f_dim * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 2, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(f_dim * 2, f_dim * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 4, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(f_dim * 4, 1,
                      kernel_size=4, stride=2, padding=1),
            # nn.Sigmoid()
            nn.InstanceNorm2d(1, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(1, 1,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(1, 1,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(1, 1,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(1, 2,
                      kernel_size=(5, 8), stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, seg):
        d0 = self.conv0(seg)
        d1 = self.conv1(d0)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        # d5 = self.conv5(d4)
        # print(np.shape(d4))
        # d6 = self.conv6(d5)
        # d7 = self.conv7(d6)
        disc = d4.view([1, 480])
        return disc
