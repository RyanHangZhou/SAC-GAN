import torch
import torch.nn as nn


class E_Transformation(nn.Module):

    def __init__(
        self,
        f_dim = 32,
        ):
        super(E_Transformation, self).__init__()

        self.f_dim = f_dim

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, self.f_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.f_dim, self.f_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Sequential(
            nn.Conv2d(self.f_dim, 4,
                      kernel_size=1, stride=1, padding=0),
        )
        self.fc_logvar = nn.Sequential(
            nn.Conv2d(self.f_dim, 4,
                      kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        e0 = self.conv0(x.reshape([-1, 6, 1, 1]))
        mu = self.fc_mu(e0)
        logvar = self.fc_logvar(e0)
        return mu, logvar


class E_Layout(nn.Module):
    
    def __init__(
        self,
        sem_len = 128,
        nClass = 19,
        f_dim = 32,
        ):
        super(E_Layout, self).__init__()

        self.sem_len = sem_len
        self.nClass = nClass
        self.f_dim = f_dim

        self.conv0 = nn.Sequential(
            nn.Conv2d(self.nClass + 4, self.f_dim,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.f_dim, self.f_dim * 2,
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
            nn.Conv2d(self.f_dim * 4, self.f_dim * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.f_dim * 8, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.f_dim * 8, self.f_dim * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.f_dim * 8, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(self.f_dim * 8, self.f_dim * 16,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.sem_len, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(self.f_dim * 16, self.f_dim * 32,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.sem_len, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(self.f_dim * 32, self.sem_len,
                      kernel_size=(4, 7), stride=1, padding=0),
        )
        self.reconz = nn.Sequential(
            nn.Dropout2d(p=0.4),
            nn.Conv2d(self.sem_len, 4,
                      kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        e0 = self.conv0(x)
        e1 = self.conv1(e0)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        e6 = self.conv6(e5)
        y = self.conv7(e6)
        z = self.reconz(y)
        return y, z
