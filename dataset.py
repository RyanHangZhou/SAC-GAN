import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import glob
import imageio
from PIL import Image
import numpy as np
from skimage.transform import resize


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, object_image_path, object_mask_path, background_image_path, semantic_label_path, select_num, transform=None):

        self.object_image_path = object_image_path
        self.object_mask_path = object_mask_path
        self.background_image_path = background_image_path
        self.semantic_label_path = semantic_label_path
        self.select_num = select_num
        self.transform = transform
        self.semantic_label_list = [x.split('/')[-1] for x in glob.glob(semantic_label_path + '/*')]
        assert len(self.semantic_label_path) != 0, "semantic_label_path is empty"
        self.nClass = 30
        self.re_h = 540
        self.re_w = 960

    def __len__(self):
        return len(self.select_num)

    def __getitem__(self, idx):
        idx = int(idx)
        idx = self.select_num[idx]

        object_image = imageio.imread(os.path.join(self.object_image_path, self.semantic_label_list[idx][:-15]+'.png'))
        object_mask = imageio.imread(os.path.join(self.object_mask_path, self.semantic_label_list[idx]))
        background_image = imageio.imread(os.path.join(self.background_image_path, self.semantic_label_list[idx][:-15]+'.png'))
        layout = self.get_seg_channels(os.path.join(self.semantic_label_path, self.semantic_label_list[idx]))

        idx_another = np.mod(idx+1, len(self.select_num))
        cond_object_mask = imageio.imread(os.path.join(self.object_mask_path, self.semantic_label_list[idx_another]))
        cond_layout = self.get_seg_channels(os.path.join(self.semantic_label_path, self.semantic_label_list[idx_another]))

        object_image = resize(object_image, (self.re_h, self.re_w))
        object_mask = resize(object_mask, (self.re_h, self.re_w))
        layout = resize(layout, (self.re_h, self.re_w))
        background_image = resize(background_image, (self.re_h, self.re_w))
        cond_object_mask = resize(cond_object_mask, (self.re_h, self.re_w))
        cond_layout = resize(cond_layout, (self.re_h, self.re_w))

        sample = {'object_image': object_image, 'object_mask': object_mask, 'background_image': background_image, \
                  'layout': layout, 'cond_object_mask': cond_object_mask, 'cond_layout': cond_layout, \
                  'semantic_label_path': self.semantic_label_list[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_seg_channels(self, seg_path):
        seg_ = Image.open(seg_path)
        ignore_label = 0
        chn_and_segID = {-1: 22, 0: ignore_label, 1: 1, 2: ignore_label,
                        3: ignore_label, 4: ignore_label, 5: 2, 6: 3,
                        7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 10,
                        14: 11, 15: 12, 16: 13, 17: 14,
                        18: 14, 19: 15, 20: 16, 21: 17, 22: 18, 23: 19, 24: 20, 25: 21, 26: 22,
                        27: 23, 28: 24, 29: 25, 30: 26, 31: 27, 32: 28, 33: 29}
        seg_ = np.array(seg_)
        seg = seg_.copy()
        for k, v in chn_and_segID.items():
            seg[seg_ == k] = v

        # change 1 channel seg map to n channel binary maps
        mask = np.zeros((seg.shape[0], seg.shape[1], self.nClass), np.uint8)
        for n in range(self.nClass):
            mask[seg == n, n] = 255
        return mask

class ToTensor():
    def __call__(self, sample):
        object_image, object_mask = sample['object_image'], sample['object_mask']
        background_image, layout = sample['background_image'], sample['layout']
        cond_object_mask, cond_layout = sample['cond_object_mask'], sample['cond_layout']
        semantic_label_path = sample['semantic_label_path']

        object_image = np.transpose(object_image, [2, 0, 1]).astype(np.float32)
        object_mask = object_mask[None, None, :, :].astype(np.float32)
        background_image = np.transpose(background_image, [2, 0, 1]).astype(np.float32)
        layout = np.transpose(layout, [2, 0, 1]).astype(np.float32)
        cond_object_mask = cond_object_mask[None, None, :, :].astype(np.float32)
        cond_layout = np.transpose(cond_layout, [2, 0, 1]).astype(np.float32)

        return (torch.from_numpy(object_image), torch.from_numpy(object_mask), torch.from_numpy(background_image), \
            torch.from_numpy(layout), torch.from_numpy(cond_object_mask), torch.from_numpy(cond_layout), semantic_label_path)


class Reparameterize():
    def __call__(self, mu, logvar, mode):
        if mode == 'train':
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu


class Pad_to_nClass():
    def __call__(self, x):
        self.nClass = 30
        self.re_h = 540
        self.re_w = 960
        target_channel = 22
        pad_small = Variable(torch.zeros(1, self.nClass - 1, self.re_h, self.re_w)).cuda()
        pad_before_small = Variable(torch.zeros(1, target_channel, self.re_h, self.re_w)).cuda()
        pad_after_small = Variable(torch.zeros(1, self.nClass - target_channel - 1, self.re_h, self.re_w)).cuda()

        if target_channel == 0:
            padded = torch.cat((x, pad_small), 1)
        elif target_channel == (self.nClass - 1):
            padded = torch.cat((pad_small, x), 1)
        else:
            padded = torch.cat((pad_before_small, x, pad_after_small), 1)

        return padded

