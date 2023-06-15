import torch
import torch.nn as nn


class D_Transformation(nn.Module):

    def __init__(
        self,
        f_dim = 32,
        ):
        super(D_Transformation, self).__init__()

        self.f_dim = f_dim

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, self.f_dim, kernel_size=1),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.f_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e0 = self.conv0(x.reshape([-1, 6, 1, 1]))
        disc = self.conv1(e0)
        return disc


class D_Layout(nn.Module):
    
    def __init__(
        self,
        class_num = 19,
        f_dim = 8,
        ):
        super(D_Layout, self).__init__()

        self.class_num = class_num
        self.f_dim = f_dim

        self.conv0 = nn.Sequential(
            nn.Conv2d((class_num + 1), self.f_dim * 1,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.f_dim * 1, self.f_dim * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.f_dim * 2, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.f_dim * 2, self.f_dim * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.f_dim * 4, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.f_dim * 4, 1,
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
        b = seg.shape[0]
        d0 = self.conv0(torch.cat([seg, mask], 1))
        d1 = self.conv1(d0)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        d5 = self.conv5(d4)
        d6 = self.conv6(d5)
        d7 = self.conv7(d6)
        disc = d7.view([b, 2])
        return disc

