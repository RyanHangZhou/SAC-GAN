import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from utils import misc

class SACGAN(nn.Module):

    def __init__(
        self,
        e_trans,
        e_patch, 
        e_layout, 
        d_trans,
        d_layout,
        args, 
        ):
        super().__init__()

        self.target_class = args.target_class
        self.patch_s = args.patch_s
        self.img_h = args.img_h
        self.img_w = args.img_w
        self.class_num = args.class_num
        self.layout_dim = args.layout_dim
        self.object_dim = args.object_dim
        self.device = args.device

        if self.target_class == 'car':
            self.target_label = 13
        elif self.target_class == 'truck':
            self.target_label = 14
        elif self.target_class == 'bus':
            self.target_label = 15
        elif self.target_class == 'person':
            self.target_label = 11

        """ Networks """
        self.e_trans = e_trans
        self.e_patch = e_patch
        self.e_layout = e_layout
        self.d_trans = d_trans
        self.d_layout = d_layout

        self.fc_loc = nn.Sequential(
            nn.Linear(self.layout_dim + self.object_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 5)
        )
        self.rotFix = Variable(torch.from_numpy(np.array([1., 0., 1., 0., 1.])).float()).to(self.device)

        """ Init """
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0], dtype=torch.float))

        """ Define Loss """
        self.BCELoss = nn.BCELoss().to(self.device)


    def forward(self, target): 
        theta_gt    = target['theta_gt'].to(self.device)
        layout      = target['layout'].to(self.device)
        cond_layout = target['cond_layout'].to(self.device)
        object_mask = target['object_mask'].to(self.device)
        obj         = target['norm_mask'].to(self.device)
        edge        = target['object_edge'].to(self.device)

        """ 1. reconstruct theta from theta_gt """
        z_mu, z_logvar = self.e_trans(theta_gt)
        z_reparam = self.re_param(z_mu, z_logvar, mode='train')
        theta_rec = self.stn(z_reparam, layout, obj, edge)[0]

        b = target['layout'].shape[0]
        z = torch.FloatTensor(b, 4, 1, 1).normal_(0, 1)
        z = Variable(z).to(self.device)

        """ 2. generate theta from z """
        theta_gen, object_mask_gen = self.stn(z, layout, obj, edge)
        cond_theta_gen, con_obj_mask_gen = self.stn(z, cond_layout, obj, edge)

        # obtain layout
        layout_gt = layout * (1. - object_mask) + self.pad_to_nclass(object_mask)
        layout_gen = layout * (1. - object_mask_gen) + self.pad_to_nclass(object_mask_gen)
        layout_ref_gen = cond_layout * (1. - con_obj_mask_gen) + self.pad_to_nclass(con_obj_mask_gen)

        """ 3. reconstruction loss """
        kl_loss = torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mu ** 2 - z_logvar.exp(), dim = 1), dim = 0)
        theta_rec_loss = torch.mean(torch.abs(theta_rec - theta_gt))
        rec_loss = theta_rec_loss + 0.01*kl_loss
    
        """ 4. GAN loss """
        # T discriminator
        d_theta_gt = self.d_trans(theta_gt)
        d_theta_rec = self.d_trans(theta_rec)
        d_theta_gen = self.d_trans(theta_gen)
        d_theta_gen_ref = self.d_trans(cond_theta_gen)

        # T loss
        true_tensor = Variable(torch.FloatTensor(d_theta_gt.data.size()).fill_(1.)).to(self.device)
        fake_tensor = Variable(torch.FloatTensor(d_theta_gt.data.size()).fill_(0.)).to(self.device)
        d_t_loss = self.BCELoss(d_theta_gt, true_tensor) + self.BCELoss(d_theta_rec, fake_tensor) + \
                     self.BCELoss(d_theta_gen, fake_tensor) + self.BCELoss(d_theta_gen_ref, fake_tensor)
        g_t_loss = self.BCELoss(d_theta_rec, true_tensor) + self.BCELoss(d_theta_gen, true_tensor) + self.BCELoss(d_theta_gen_ref, true_tensor)

        # layout discriminator
        d_layout_gt = self.d_layout(layout_gt, object_mask)
        d_layout_gen = self.d_layout(layout_gen, object_mask_gen)
        d_layout_ref_gen = self.d_layout(layout_ref_gen, con_obj_mask_gen)

        # layout loss
        true_tensor = Variable(torch.FloatTensor(d_layout_gt.data.size()).fill_(1.)).to(self.device)
        fake_tensor = Variable(torch.FloatTensor(d_layout_gt.data.size()).fill_(0.)).to(self.device)
        d_layout_loss = self.BCELoss(d_layout_gt, true_tensor) + self.BCELoss(d_layout_gen, fake_tensor) + self.BCELoss(d_layout_ref_gen, fake_tensor)
        g_layout_loss = self.BCELoss(d_layout_gen, true_tensor) + self.BCELoss(d_layout_ref_gen, true_tensor)

        loss = {}
        loss["rec_loss"] = rec_loss
        loss["d_t_loss"] = d_t_loss
        loss["g_t_loss"] = g_t_loss
        loss["d_layout_loss"] = d_layout_loss
        loss["g_layout_loss"] = g_layout_loss

        return loss


    def re_param(self, mu, logvar, mode): 
        if mode == 'train':
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu


    def stn(self, z, layout, obj, edge):
        # predict theta (bX2X3)
        theta = self.localize(z, layout, obj, edge)

        h_img, w_img = layout.size()[2:]
        h_obj, w_obj = obj.size()[2:]

        padded = nn.ZeroPad2d((int((w_img-w_obj)/2), w_img-w_obj-int((w_img-w_obj)/2), int((h_img-h_obj)/2), h_img-h_obj-int((h_img-h_obj)/2)))
        obj = padded(obj)

        grid = F.affine_grid(theta, obj.size(), align_corners=True) # generate a 2D flow field (sampling grid), given a batch of affine matrices theta
        obj = F.grid_sample(obj, grid, align_corners=True)

        return theta, obj


    def localize(self, z, layout, obj_mask, edge_obj):
        b = layout.shape[0]

        # prepare network inputs
        z = z.expand([b, -1, self.img_h, self.img_w])
        layout_z = torch.cat([layout, z], 1)
        mask_edge = torch.cat([obj_mask, edge_obj], 1)

        # encode inputs
        layout_code = self.e_layout(layout_z)[0]
        layout_code = layout_code.view(-1, self.layout_dim)
        object_code = self.e_patch(mask_edge)[0].view(-1, self.object_dim)
        code = torch.cat((layout_code, object_code), -1)

        # predict theta
        theta = self.fc_loc(code).view(b, -1)

        # shuffle theta into 2X3 2D matrices
        rotFix = torch.unsqueeze(self.rotFix, 0).repeat(b, 1)
        theta = theta * rotFix
        theta = torch.cat((theta[:, 0:4], torch.unsqueeze(theta[:, 0], 1), torch.unsqueeze(theta[:, 4], 1)), 1)
        theta = theta.view(-1, 2, 3)

        return theta


    def pad_to_nclass(self, x):
        b = x.shape[0]
        pad_small = Variable(torch.zeros(b, self.class_num - 1, self.img_h, self.img_w)).to(self.device)
        pad_before_small = Variable(torch.zeros(b, self.target_label, self.img_h, self.img_w)).to(self.device)
        pad_after_small = Variable(torch.zeros(b, self.class_num - self.target_label - 1, self.img_h, self.img_w)).to(self.device)

        if self.target_label == 0:
            padded = torch.cat((x, pad_small), 1)
        elif self.target_label == (self.class_num - 1):
            padded = torch.cat((pad_small, x), 1)
        else:
            padded = torch.cat((pad_before_small, x, pad_after_small), 1)

        return padded


    def compose(self, obj, mask, theta, image, folder_name, result_dir, indx):
        h_img, w_img = image.size()[2:]
        h_obj, w_obj = obj.size()[2:]

        padded = nn.ZeroPad2d((int((w_img-w_obj)/2), w_img-w_obj-int((w_img-w_obj)/2), int((h_img-h_obj)/2), h_img-h_obj-int((h_img-h_obj)/2)))
        obj = padded(obj).to(dtype=torch.float64)
        mask = padded(mask).to(dtype=torch.float64)

        grid = F.affine_grid(theta, mask.size(), align_corners=True).to(dtype=torch.float64)
        obj = F.grid_sample(obj, grid, align_corners=True)
        mask = F.grid_sample(mask, grid, align_corners=True)

        composed_img = mask * obj+(1-mask)*image

        misc.save_img(obj[0], result_dir, 'test/object_image', indx)
        misc.save_img(mask[0], result_dir, 'test/mask_image', indx)
        misc.save_img(composed_img[0], result_dir, 'test/image', indx)


    def inference(self, target, result_dir):
        background_image = target['background_image'].to(self.device)
        layout           = target['layout'].to(self.device)
        object_mask      = target['object_mask'].to(self.device)
        patch_obj        = target['patch_obj'].to(self.device)
        patch_mask        = target['patch_mask'].to(self.device)
        obj              = target['norm_mask'].to(self.device)
        edge             = target['object_edge'].to(self.device)
        sample           = target['sample']

        """ 1. z -> T_gen """
        b = target['layout'].shape[0]
        z = torch.FloatTensor(b, 4, 1, 1).normal_(0, 1)
        z = Variable(z).to(self.device)
        theta_gen, object_mask_gen = self.stn(z, layout, obj, edge)

        """ 2. Object composition """
        local_h, local_w = patch_obj.shape[2:]
        theta_gen[0, 0, 0] = theta_gen[0, 0, 0] * (local_w/self.patch_s)
        theta_gen[0, 1, 1] = theta_gen[0, 1, 1] * (local_h/self.patch_s)
        theta_gen[0, 0, 2] = theta_gen[0, 0, 2] * (local_w/self.patch_s)
        theta_gen[0, 1, 2] = theta_gen[0, 1, 2] * (local_h/self.patch_s)

        self.compose(patch_obj, patch_mask, theta_gen, background_image, 'test/image', result_dir, sample[0])
        misc.save_layout(layout, result_dir, 'test/GT_layout', sample[0])
        layout_gen = layout * (1. - object_mask_gen) + self.pad_to_nclass(object_mask_gen)
        misc.save_layout(layout_gen, result_dir, 'test/layout', sample[0])

