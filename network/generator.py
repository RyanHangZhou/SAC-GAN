import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
from utils import util

from torchvision.utils import save_image
from collections import namedtuple

from network import ResNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sem_len = 128
obj_len = 30
nClass = 19 # 30

class STNet(nn.Module):
    def __init__(self):
        super(STNet, self).__init__()

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(sem_len+obj_len, 32),
            nn.ReLU(True),
            nn.Linear(32, 5)
        )

        # initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0], dtype=torch.float))

        ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
        resnet_model = Where_Encoder()
        layout_decoder = Layout_Reconstructor()
        object_decoder = Object_Reconstructor()
        self.resnet_model = resnet_model.to(device)
        self.layout_decoder = layout_decoder.to(device)
        self.object_decoder = object_decoder.to(device)

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
        feat_sem, z_dec = self.resnet_model(sem_img) # basically an encoder network (HXWX3->90X1)
        layout_dec = self.layout_decoder(feat_sem)
        feat_sem = feat_sem.view(-1, sem_len)
        feat_obj, _ = self.resnet_model_mask(obj_mask)
        feat_obj_ = torch.unsqueeze(feat_obj, 2)
        feat_obj_ = torch.unsqueeze(feat_obj_, 2)
        object_dec = self.object_decoder(feat_obj_)
        feat_obj = feat_obj.view(-1, obj_len)
        feat = torch.cat((feat_sem, feat_obj), -1)

        theta = self.fc_loc(feat) # still an encoder network (90X1->6X1)
        
        theta = theta * self.rotFix # fix rotation
        theta = torch.cat((theta[:, 0:4], torch.unsqueeze(theta[:, 0],0), torch.unsqueeze(theta[:, 4],0)), 1)
        theta = theta.view(-1, 2, 3) # reshape (6X1->2X3)

        return theta, z_dec, layout_dec, object_dec

    # Spatial transformer network forward function
    def stn(self, sem_img, obj_mask, edge_obj, results_path, indx):
        theta, z_dec, layout_dec, object_dec = self.localization(sem_img, torch.cat([obj_mask, edge_obj], 1))

        h_img, w_img = sem_img.size()[2], sem_img.size()[3]
        h_obj, w_obj = obj_mask.size()[2], obj_mask.size()[3]

        padded = nn.ZeroPad2d((int((w_img-w_obj)/2), w_img-w_obj-int((w_img-w_obj)/2), int((h_img-h_obj)/2), h_img-h_obj-int((h_img-h_obj)/2)))
        obj_mask = padded(obj_mask)

        grid = F.affine_grid(theta, obj_mask.size(), align_corners=True) # Generates a 2D flow field (sampling grid), given a batch of affine matrices theta
        obj_mask = F.grid_sample(obj_mask, grid, align_corners=True)

        return theta, obj_mask, grid, z_dec, layout_dec, object_dec

    def object_compose(self, obj, mask, theta, image, directory_name, results_path, indx, is_train=False):
        h_img, w_img = image.size()[2], image.size()[3]
        h_obj, w_obj = obj.size()[2], obj.size()[3]

        padded = nn.ZeroPad2d((int((w_img-w_obj)/2), w_img-w_obj-int((w_img-w_obj)/2), int((h_img-h_obj)/2), h_img-h_obj-int((h_img-h_obj)/2)))
        obj = padded(obj).to(dtype=torch.float64)
        mask = padded(mask).to(dtype=torch.float64)

        grid = F.affine_grid(theta, mask.size(), align_corners=True).to(dtype=torch.float64)
        obj = F.grid_sample(obj, grid, align_corners=True)
        mask = F.grid_sample(mask, grid, align_corners=True)

        composed_img = self.compose(obj, mask, image)

        for i in range(len(indx)):
            if is_train:
                util.save_img(obj[0], results_path, 'train_obj', indx)
                util.save_img(mask[0], results_path, 'train_mask', indx)
                util.save_img(composed_img[0], results_path, 'train_composed'+directory_name, indx)
            else:
                util.save_img(obj[0], results_path, 'test_obj', indx)
                util.save_img(mask[0], results_path, 'test_mask', indx)
                util.save_img(composed_img[0], results_path, 'test_composed'+directory_name, indx)

            # save_image(obj[i], os.path.join(results_path, 'obj'+indx[i]))
            # save_image(mask[i], os.path.join(results_path, 'mask'+indx[i]))
            # save_image(composed_img[i], os.path.join(results_path, 'composed'+indx[i]))

    
    def compose(self, obj, mask, background): 
        # composed_img = mask * obj+(1-mask)*background
        # print(torch.max(mask))
        # print(torch.max(obj))
        # print(torch.max(background))
        # sss
        mask = mask/255.
        composed_img = mask * obj+(1-mask)*background
        return composed_img

    def forward(self, sem_img, obj_mask, edge_obj, results_path, indx):
        # transform the input
        theta, obj_mask, grid, z_dec, layout_dec, object_dec = self.stn(sem_img, obj_mask, edge_obj, results_path, indx)
        # composed_img = self.compose(obj, mask, background)

        return theta, grid, obj_mask, z_dec, layout_dec, object_dec

# T encoder
class TEncoderNet(nn.Module):
    def __init__(self):
        super(TEncoderNet, self).__init__()

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
        e0 = self.conv0(x.reshape([-1, 6, 1, 1]))
        mu = self.fc_mu(e0)
        logvar = self.fc_logvar(e0)

        return mu, logvar


class Where_Encoder(nn.Module):
    def __init__(self):
        super(Where_Encoder, self).__init__()

        f_dim = 32
        self.conv0 = nn.Sequential(
            nn.Conv2d(nClass + 4, f_dim,
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
            nn.Conv2d(f_dim * 8, f_dim * 16,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(sem_len, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(f_dim * 16, f_dim * 32,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(sem_len, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(f_dim * 32, sem_len,
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

class Layout_Reconstructor(nn.Module):
    def __init__(self):
        super(Layout_Reconstructor, self).__init__()

        f_dim = 8
        self.convT0 = nn.Sequential(
            nn.ConvTranspose2d(sem_len, f_dim * 8,
                               kernel_size=(4, 7), stride=1, padding=0),
            nn.InstanceNorm2d(f_dim * 8, affine=False),
            nn.ReLU(),
        )
        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(f_dim * 8, f_dim * 8,
                               kernel_size=4, stride=2, padding=1, output_padding=(0,1)),
            nn.InstanceNorm2d(f_dim * 8, affine=False),
            nn.ReLU(),
        )
        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(f_dim * 8, f_dim * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 4, affine=False),
            nn.ReLU(),
        )
        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(f_dim * 4, f_dim * 2,
                               kernel_size=4, stride=2, padding=1, output_padding=(1,0)),
            nn.InstanceNorm2d(f_dim * 2, affine=False),
            nn.ReLU(),
        )
        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(f_dim * 2, f_dim * 1,
                               kernel_size=4, stride=2, padding=1, output_padding=(1,0)),
            nn.InstanceNorm2d(f_dim * 1, affine=False),
            nn.ReLU(),
        )
        self.convT5 = nn.Sequential(
            nn.ConvTranspose2d(f_dim * 1, f_dim * 1,
                               kernel_size=4, stride=2, padding=1, output_padding=(1,0)),
            nn.InstanceNorm2d(f_dim * 1, affine=False),
            nn.ReLU(),
        )
        self.convT6 = nn.Sequential(
            nn.ConvTranspose2d(f_dim * 1, f_dim * 1,
                               kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 1, affine=False),
            nn.ReLU(),
        )
        self.convT7 = nn.Sequential(
            nn.ConvTranspose2d(f_dim * 1, nClass,
                               kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, em):
        m0 = self.convT0(em)
        m1 = self.convT1(m0)
        m2 = self.convT2(m1)
        m3 = self.convT3(m2)
        m4 = self.convT4(m3)
        m5 = self.convT5(m4)
        m6 = self.convT6(m5)
        mask = self.convT7(m6)

        return mask

class Object_Reconstructor(nn.Module): 
    def __init__(self):
        super(Object_Reconstructor, self).__init__()

        f_dim = 8
        self.convT0 = nn.Sequential(
            nn.ConvTranspose2d(obj_len, f_dim * 8,
                               kernel_size=4, stride=1, padding=0),
            nn.InstanceNorm2d(f_dim * 8, affine=False),
            nn.ReLU(),
        )
        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(f_dim * 8, f_dim * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 8, affine=False),
            nn.ReLU(),
        )
        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(f_dim * 4, f_dim * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 4, affine=False),
            nn.ReLU(),
        )
        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(f_dim * 2, f_dim * 1,
                               kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 2, affine=False),
            nn.ReLU(),
        )
        # self.convT4 = nn.Sequential(
        #     nn.ConvTranspose2d(f_dim * 2, f_dim * 1,
        #                        kernel_size=4, stride=2, padding=1),
        #     nn.InstanceNorm2d(f_dim * 1, affine=False),
        #     nn.ReLU(),
        # )
        self.convT7 = nn.Sequential(
            nn.ConvTranspose2d(f_dim * 1, 2,
                               kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, em):
        m0 = self.convT0(em)
        m1 = self.convT1(m0)
        m2 = self.convT2(m1)
        m3 = self.convT3(m2)
        # m4 = self.convT4(m3)
        mask = self.convT7(m3)

        return mask


