import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os

from torchvision.utils import save_image
from collections import namedtuple

import ResNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sem_len = 128
obj_len = 30
nClass = 29+1

class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(sem_len+obj_len, 32),
            nn.ReLU(True),
            nn.Linear(32, 5)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0], dtype=torch.float))

        ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
        resnet_model = Where_Encoder()
        self.resnet_model = resnet_model.to(device)

        resnet18_config_mask = ResNetConfig(block = ResNet.BasicBlock, 
            n_blocks = [2, 2, 2, 2], channels = [16, 32, 64, 128])
        resnet34_config_mask = ResNetConfig(block = ResNet.BasicBlock,
            n_blocks = [3, 4, 6, 3], channels = [64, 128, 256, 512])
        resnet50_config_mask = ResNetConfig(block = ResNet.Bottleneck,
            n_blocks = [3, 4, 6, 3], channels = [64, 128, 256, 512])
        resnet101_config = ResNetConfig(block = ResNet.Bottleneck,
            n_blocks = [3, 4, 23, 3], channels = [64, 128, 256, 512])
        resnet_model_mask = ResNet.ResNet(resnet34_config_mask, 2, obj_len)
        self.resnet_model_mask = resnet_model_mask.to(device)

        # Rotation fixed
        self.rotFix = Variable(torch.from_numpy(np.array([1., 0., 1., 0., 1.])).float()).cuda()

    def localization(self, sem_img, obj_mask):
        feat_sem, _ = self.resnet_model(sem_img) # basically an encoder network (HXWX3->90X1)
        feat_sem = feat_sem.view(-1, sem_len)
        feat_obj, _ = self.resnet_model_mask(obj_mask)
        feat_obj = feat_obj.view(-1, obj_len)
        feat = torch.cat((feat_sem, feat_obj), -1)

        theta = self.fc_loc(feat) # still an encoder network (90X1->6X1)
        theta = theta * self.rotFix # fix rotation
        theta = torch.cat((theta[:, 0:4], torch.unsqueeze(theta[:, 0],0), torch.unsqueeze(theta[:, 4],0)), 1)
        theta = theta.view(-1, 2, 3) # reshape (6X1->2X3)

        return theta

    # Spatial transformer network forward function
    def stn(self, sem_img, obj_mask, edge_obj, results_path, indx):
        theta = self.localization(sem_img, torch.cat([obj_mask, edge_obj], 1))

        h_img, w_img = sem_img.size()[2], sem_img.size()[3]
        h_obj, w_obj = obj_mask.size()[2], obj_mask.size()[3]

        padded = nn.ZeroPad2d((int((w_img-w_obj)/2), w_img-w_obj-int((w_img-w_obj)/2), int((h_img-h_obj)/2), h_img-h_obj-int((h_img-h_obj)/2)))
        obj_mask = padded(obj_mask)

        grid = F.affine_grid(theta, obj_mask.size()) # Generates a 2D flow field (sampling grid), given a batch of affine matrices theta
        obj_mask = F.grid_sample(obj_mask, grid)   

        return theta, obj_mask, grid

    def object_compose(self, obj, mask, theta, image, results_path, indx):
        h_img, w_img = image.size()[2], image.size()[3]
        h_obj, w_obj = obj.size()[2], obj.size()[3]

        padded = nn.ZeroPad2d((int((w_img-w_obj)/2), w_img-w_obj-int((w_img-w_obj)/2), int((h_img-h_obj)/2), h_img-h_obj-int((h_img-h_obj)/2)))
        obj = padded(obj).to(dtype=torch.float64)
        mask = padded(mask).to(dtype=torch.float64)

        grid = F.affine_grid(theta, mask.size()).to(dtype=torch.float64)
        obj = F.grid_sample(obj, grid)
        mask = F.grid_sample(mask, grid)

        composed_img = self.compose(obj, mask, image)

        for i in range(len(indx)):
            save_image(obj[i], os.path.join(results_path, 'obj'+indx[i]))
            save_image(mask[i], os.path.join(results_path, 'mask'+indx[i]))
            save_image(composed_img[i], os.path.join(results_path, 'composed'+indx[i]))

    
    def compose(self, obj, mask, background): 
        composed_img = mask * obj+(1-mask)*background
        return composed_img

    def forward(self, sem_img, obj_mask, edge_obj, results_path, indx):
        # transform the input
        theta, obj_mask, grid = self.stn(sem_img, obj_mask, edge_obj, results_path, indx)
        # composed_img = self.compose(obj, mask, background)

        return theta, grid, obj_mask


class DiscriminatorSTN(nn.Module):
    def __init__(self):
        super(DiscriminatorSTN, self).__init__()

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
        e0 = self.conv0(x.view([-1, 6, 1, 1]))
        disc = self.conv1(e0)

        return disc

class DiscriminatorWhere(nn.Module):
    def __init__(self):
        super(DiscriminatorWhere, self).__init__()

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
            nn.Sigmoid()
        )

    def forward(self, seg, mask):
        d0 = self.conv0(torch.cat([seg, mask], 1))
        d1 = self.conv1(d0)
        d2 = self.conv2(d1)
        disc = self.conv3(d2)
        return disc


class Where_Encoder_Sup(nn.Module):
    def __init__(self):
        super(Where_Encoder_Sup, self).__init__()

        f_dim = 32
        self.conv0 = nn.Sequential(
            nn.Conv2d(6, f_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(f_dim, f_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Sequential(
            nn.Conv2d(f_dim, 4,
                      kernel_size=1, stride=1, padding=0),
        )
        self.fc_logvar = nn.Sequential(
            nn.Conv2d(f_dim, 4,
                      kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        e0 = self.conv0(x.view([-1, 6, 1, 1]))
        mu = self.fc_mu(e0)
        logvar = self.fc_logvar(e0)

        return mu, logvar

class Where_Encoder(nn.Module):
    def __init__(self):
        super(Where_Encoder, self).__init__()

        f_dim = 32
        self.conv0 = nn.Sequential(
            nn.Conv2d(30 + 4, f_dim,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(f_dim, f_dim * 2,
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
            nn.Conv2d(f_dim * 4, f_dim * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 8, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(f_dim * 8, f_dim * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 8, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(f_dim * 8, sem_len,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(sem_len, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(sem_len, sem_len,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(sem_len, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(sem_len, sem_len,
                      kernel_size=(4, 7), stride=1, padding=0),
        )
        self.reconz = nn.Sequential(
            nn.Dropout2d(p=0.4),
            nn.Conv2d(sem_len, 4,
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

