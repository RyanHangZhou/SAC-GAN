# -*- coding: utf-8 -*-
from __future__ import print_function


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.utils import save_image

from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import argparse

import copy
from collections import namedtuple
import os
import random
import shutil
import time

import dataset
import ResNet
import SACGAN
import utils
import edge


# SEED = 1234
SEED = int(time.time())
beta1 = 0.5
nClass = 30

results_path = 'results/aachen'
checkpoint_path = 'checkpoints'


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


train_transform = transforms.Compose([
dataset.ToTensor(),
])

eval_transform = transforms.Compose([
dataset.ToTensor()
])

BATCH_SIZE = 1
EPOCHS = 100
flag_aspect = 1 # keep ratio aspect (flag=1)

object_image_path = 'data/aachen/object_image'
object_mask_path = 'data/aachen/object_mask'
background_image_path = 'data/aachen/background_image'
semantic_label_path = 'data/aachen/semantic_label'

image_num = len([lists for lists in os.listdir(semantic_label_path) if os.path.isfile(os.path.join(semantic_label_path, lists))])
train_num = int(0.7*image_num)
test_num = int(0.3*image_num)
valid_num = test_num
image_perm = random.sample(range(image_num), k=image_num)
image_train = image_perm[0:train_num - 1]
image_test = image_perm[train_num:train_num  + test_num - 1]
image_valid = image_test

CUDA = True
kwargs = {'num_workers': 4,'pin_memory': True} if CUDA else {}

train_iterator = torch.utils.data.DataLoader(dataset.ImageDataset(object_image_path, object_mask_path, background_image_path, 
    semantic_label_path, image_train, transform=train_transform),
    batch_size=BATCH_SIZE, shuffle=False, **kwargs, drop_last=True)
valid_iterator = torch.utils.data.DataLoader(dataset.ImageDataset(object_image_path, object_mask_path, background_image_path, 
    semantic_label_path, image_valid, transform=eval_transform),
    batch_size=BATCH_SIZE, shuffle=False, **kwargs, drop_last=True)
test_iterator = torch.utils.data.DataLoader(dataset.ImageDataset(object_image_path, object_mask_path, background_image_path, 
    semantic_label_path, image_test, transform=eval_transform),
    batch_size=BATCH_SIZE, shuffle=False, **kwargs, drop_last=True)


cond_stn_model = SACGAN.STN()
disSTN = SACGAN.DiscriminatorSTN()
disWhere = SACGAN.DiscriminatorWhere()
where_encoder_sup = SACGAN.Where_Encoder_Sup()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(cond_stn_model):,} trainable parameters')

START_LR = 2e-4

optimizer = optim.Adam(list(where_encoder_sup.parameters())+list(cond_stn_model.parameters()), lr=START_LR, betas=(beta1, 0.999))
optimizer_d = optim.Adam(list(disSTN.parameters())+list(disWhere.parameters()), lr=START_LR, betas=(0.5, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.75, last_epoch=-1)
scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, 30, gamma=0.75, last_epoch=-1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cond_stn_model = cond_stn_model.to(device)
disSTN = disSTN.to(device)
disWhere = disWhere.to(device)
where_encoder_sup = where_encoder_sup.to(device)

class IteratorWrapper:
    def __init__(self, iterator):
        self.iterator = iterator
        self._iterator = iter(iterator)

    def __next__(self):
        try:
            object_image, object_mask, background_image, layout, cond_object_mask, cond_layout, indx = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterator)
            object_image, object_mask, background_image, layout, cond_object_mask, cond_layout, indx, *_ = next(self._iterator)

        return object_image, object_mask, background_image, layout, cond_object_mask, cond_layout, indx

    def get_batch(self):
        return next(self)


def train(cond_stn_model, where_encoder_sup, disSTN, disWhere, iterator, optimizer, optimizer_d, scheduler, scheduler_d, device, flag):
    
    epoch_loss = 0
    
    if (flag==0):
        optimizer.zero_grad()
    else:
        optimizer_d.zero_grad()
    
    for (object_image, object_mask, background_image, layout, cond_object_mask, cond_layout, indx) in iterator:

        shape = list(object_image.size())
        shape_mask = list(object_mask.size())
        object_mask = object_mask.reshape(shape_mask[0] * shape_mask[1], *shape_mask[2:])
        cond_object_mask = cond_object_mask.reshape(shape_mask[0] * shape_mask[1], *shape_mask[2:])

        shape_new = list(object_mask.size())
        rand_index = torch.randperm(shape_new[0])
        object_image = object_image[rand_index].to(device)
        object_mask = object_mask[rand_index].to(device)
        background_image = background_image[rand_index].to(device)
        layout = layout[rand_index].to(device)
        cond_object_mask = cond_object_mask[rand_index].to(device)
        cond_layout = cond_layout[rand_index].to(device)

        new_h, new_w = 64, 64
        norm_mask, norm_obj, theta_gt = utils.batch_object_extract(object_mask.detach().cpu().numpy(), object_image.detach().cpu().numpy(), new_h, new_w, indx, flag_aspect)
        _, _, con_theta_gt = utils.batch_object_extract(cond_object_mask.detach().cpu().numpy(), object_image.detach().cpu().numpy(), new_h, new_w, indx, flag_aspect)

        norm_obj = torch.tensor(norm_obj)
        edge_obj = torch.tensor(edge.compute_edge_for_input(norm_obj)).to(device)
        norm_obj = norm_obj.to(device)
        norm_mask = torch.tensor(norm_mask).to(device)
        theta_gt = torch.tensor(theta_gt).to(device)
        con_theta_gt = torch.tensor(con_theta_gt).to(device)

        # encode theta_gt
        where_sup_mu, where_sup_logvar = where_encoder_sup(theta_gt)
        reparameterize = dataset.Reparameterize()
        where_sup_reparam = reparameterize(where_sup_mu, where_sup_logvar, mode='train')
        where_sup_reparam_expand = where_sup_reparam.expand([-1, -1, shape_new[2], shape_new[3]])
        layout_ = torch.cat([layout, where_sup_reparam_expand], 1)
        where_KL_loss = torch.mean(-0.5 * torch.sum(1 + where_sup_logvar - where_sup_mu ** 2 - where_sup_logvar.exp(), dim = 1), dim = 0)

        theta, grid, obj_mask = cond_stn_model(layout_, norm_mask, edge_obj, results_path, indx)

        #real
        pad_to_class = dataset.Pad_to_nClass()
        real_bbs_pad_small = pad_to_class(cond_object_mask)
        bb_on_real_seg_small_masked = cond_layout * (1. - cond_object_mask)
        bb_on_real_seg_small = bb_on_real_seg_small_masked + real_bbs_pad_small
        #fake
        bb_padded_small = pad_to_class(obj_mask)
        bb_on_seg_small_masked = layout * (1. - obj_mask)
        bb_on_seg_small = bb_on_seg_small_masked + bb_padded_small
        # bb_on_edge_small = cond_edge_small * (1. - bb_transformed_small)

        b_z_appr = torch.FloatTensor(1, 4, 1, 1).normal_(0, 1)
        input_z_appr = Variable(b_z_appr).to(device)
        input_z_appr_expand = input_z_appr.expand([-1, -1, shape_new[2], shape_new[3]])
        layout___ = torch.cat([layout, input_z_appr_expand], 1)
        theta_fake2, _, _ = cond_stn_model(layout___, norm_mask, edge_obj, results_path, indx)

        cond_layout_ = torch.cat([cond_layout, input_z_appr_expand], 1)
        theta_fake3, _, _ = cond_stn_model(cond_layout_, norm_mask, edge_obj, results_path, indx)        

        d_stn_real = disSTN(theta_gt)
        d_stn_fake = disSTN(theta)
        d_stn_fake2 = disSTN(theta_fake2)
        d_stn_fake3 = disSTN(theta_fake3)

        rec_stn_loss = torch.mean(torch.abs(theta - theta_gt))

        criterionGAN = nn.BCELoss().cuda()
        true_tensor = Variable(torch.cuda.FloatTensor(d_stn_real.data.size()).fill_(1.))
        fake_tensor = Variable(torch.cuda.FloatTensor(d_stn_real.data.size()).fill_(0.))
        d_stn_loss = criterionGAN(d_stn_real, true_tensor) + criterionGAN(d_stn_fake, fake_tensor) + \
                     criterionGAN(d_stn_fake2, fake_tensor) + criterionGAN(d_stn_fake3, fake_tensor)
        g_stn_loss = criterionGAN(d_stn_fake, true_tensor) + criterionGAN(d_stn_fake2, true_tensor) + criterionGAN(d_stn_fake3, true_tensor)

        d_where_real = disWhere(bb_on_real_seg_small, cond_object_mask) # layout
        d_where_fake = disWhere(bb_on_seg_small, obj_mask) # layout
        true_tensor = Variable(torch.cuda.FloatTensor(d_where_real.data.size()).fill_(1.))
        fake_tensor = Variable(torch.cuda.FloatTensor(d_where_real.data.size()).fill_(0.))
        d_where_loss = criterionGAN(d_where_real, true_tensor) + criterionGAN(d_where_fake, fake_tensor)
        g_where_loss = criterionGAN(d_where_fake, true_tensor)

        if(flag==0): 
            loss = 1.5*rec_stn_loss + 1*where_KL_loss + 1*g_stn_loss+ 1*g_where_loss
        else: 
            loss = 1*d_stn_loss + 1*d_where_loss

        loss.backward()
        if(flag==0): 
            optimizer.step()
            scheduler.step()
        else:
            optimizer_d.step()
            scheduler_d.step()
        
        epoch_loss += loss.item()
        
    epoch_loss /= len(iterator)

        
    return epoch_loss


def evaluate(cond_stn_model, where_encoder_sup, disSTN, disWhere, iterator, device):
    
    epoch_loss = 0

    cond_stn_model.eval()
    disSTN.eval()
    disWhere.eval()
    where_encoder_sup.eval()
    
    with torch.no_grad():

        for (object_image, object_mask, background_image, layout, cond_object_mask, cond_layout, indx) in iterator:

            shape = list(object_image.size())
            shape_mask = list(object_mask.size())
            object_mask = object_mask.reshape(shape_mask[0] * shape_mask[1], *shape_mask[2:])

            shape_new = list(object_mask.size())
            object_image = object_image.to(device)
            background_image = background_image.to(device)
            object_mask = object_mask.to(device)
            layout = layout.to(device)
            
            new_h, new_w = 64, 64
            norm_mask, norm_obj, theta_gt = utils.batch_object_extract(object_mask.detach().cpu().numpy(), object_image.detach().cpu().numpy(), new_h, new_w, indx, flag_aspect)
            local_mask, local_obj = utils.object_extract(object_mask.detach().cpu().numpy(), object_image.detach().cpu().numpy(), flag_aspect, indx)

            _, local_h, local_w = np.shape(local_mask)

            norm_obj = torch.tensor(norm_obj)
            edge_obj = torch.tensor(edge.compute_edge_for_input(norm_obj)).to(device)
            norm_obj = norm_obj.to(device)
            norm_mask = torch.tensor(norm_mask).to(device)
            theta_gt = torch.tensor(theta_gt).to(device)

            # random z from Gaussian distribution
            b_z_appr = torch.FloatTensor(1, 4, 1, 1).normal_(0, 1)
            input_z_appr = Variable(b_z_appr).to(device)
            
            where_sup_mu, where_sup_logvar = where_encoder_sup(theta_gt)
            # reparameterize = dataset.Reparameterize()
            # input_z_appr_expand = reparameterize(where_sup_mu, where_sup_logvar, mode='test')

            input_z_appr_expand = input_z_appr.expand([-1, -1, shape_new[2], shape_new[3]])
            layout_ = torch.cat([layout, input_z_appr_expand], 1)

            # object_image*object_mask
            theta, grid, obj_mask = cond_stn_model(layout_, norm_mask, edge_obj, results_path, indx)

            theta_ = theta

            rec_stn_loss = torch.mean(torch.abs(theta - theta_gt))

            # verify if the theta is correctly calculated
            # cond_stn_model.object_compose(norm_obj, norm_mask, theta_gt, background_image, indx)

            # cond_stn_model.object_compose(norm_obj, norm_mask, theta, background_image, indx)


            local_obj = torch.tensor(local_obj).to(device)
            local_mask = torch.tensor(local_mask).to(device)
            local_obj = torch.unsqueeze(local_obj, 0)
            local_mask = torch.unsqueeze(local_mask, 0)
            theta[0, 0, 0] = theta[0, 0, 0] * (local_w/new_w)
            theta[0, 1, 1] = theta[0, 1, 1] * (local_h/new_h)
            theta[0, 0, 2] = theta[0, 0, 2] * (local_w/new_w)
            theta[0, 1, 2] = theta[0, 1, 2] * (local_h/new_h)

            cond_stn_model.object_compose(local_obj, local_mask, theta, background_image, results_path, indx)

            epoch_loss += rec_stn_loss.item()
        
        epoch_loss /= len(iterator)
        
    return epoch_loss

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


best_valid_loss = float('inf')

for epoch in range(EPOCHS):
    
    start_time = time.monotonic()
    
    # discriminator
    train_loss = train(cond_stn_model, where_encoder_sup, disSTN, disWhere, train_iterator, optimizer, optimizer_d, scheduler, scheduler_d, device, flag=1)
    # generator
    train_loss = train(cond_stn_model, where_encoder_sup, disSTN, disWhere, train_iterator, optimizer, optimizer_d, scheduler, scheduler_d, device, flag=0)
    train_loss = train(cond_stn_model, where_encoder_sup, disSTN, disWhere, train_iterator, optimizer, optimizer_d, scheduler, scheduler_d, device, flag=0)

    valid_loss = evaluate(cond_stn_model, where_encoder_sup, disSTN, disWhere, valid_iterator, device)
        
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(cond_stn_model.state_dict(), os.path.join(checkpoint_path, '11_23_1.pt'))

    end_time = time.monotonic()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'Train Loss: {train_loss:.3f}')
    print(f'Test Loss: {valid_loss:.3f}')


cond_stn_model.load_state_dict(torch.load(os.path.join(checkpoint_path, '11_23_1.pt'))) # best: 10_29_2

test_loss = evaluate(cond_stn_model, where_encoder_sup, disSTN, disWhere, test_iterator, device)
print('---------------------------------------------------------------------------------------')
print('---------------------------------------------------------------------------------------')
print('---------------------------------------------------------------------------------------')
print('---------------------------------------------------------------------------------------')
print('---------------------------------------------------------------------------------------')
print('---------------------------------------------------------------------------------------')
print('---------------------------------------------------------------------------------------')
train_loss = evaluate(cond_stn_model, where_encoder_sup, disSTN, disWhere, train_iterator, device)

print(f'Test Loss: {test_loss:.3f}')
print(f'Train Loss: {train_loss:.3f}')

