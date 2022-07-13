import torch
import imageio
import numpy as np
import os
import glob
import random
from skimage.transform import resize
from PIL import Image

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, img_h, img_w, class_num, layout_flag, is_train=True):

        self.img_h = img_h
        self.img_w = img_w
        self.class_num = class_num
        self.layout_flag = layout_flag

        if is_train:
            suffix = '_train'
        else:
            suffix = '_test'
        self.object_image_path = os.path.join(dataset, 'object_image'+suffix)
        self.object_mask_path = os.path.join(dataset, 'object_mask'+suffix)
        self.background_image_path = os.path.join(dataset, 'background_image'+suffix)
        self.semantic_label_path = os.path.join(dataset, 'semantic_label'+suffix)
        self.semantic_label_list = [x.split('/')[-1] for x in glob.glob(self.semantic_label_path + '/*')]
        self.img_list = [lists for lists in os.listdir(self.object_image_path) if os.path.isfile(os.path.join(self.object_image_path, lists))]
        self.img_index = random.sample(range(len(self.img_list)), k=len(self.img_list))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        idx = int(idx)
        idx = self.img_index[idx]

        object_image = imageio.imread(os.path.join(self.object_image_path, self.semantic_label_list[idx][:-15]+'.png'))
        object_mask = imageio.imread(os.path.join(self.object_mask_path, self.semantic_label_list[idx]))
        background_image = imageio.imread(os.path.join(self.background_image_path, self.semantic_label_list[idx][:-15]+'.png'))
        layout = self.get_channels(os.path.join(self.semantic_label_path, self.semantic_label_list[idx]))

        idx_another = np.mod(idx+1, len(self.img_list))
        cond_object_mask = imageio.imread(os.path.join(self.object_mask_path, self.semantic_label_list[idx_another]))
        cond_layout = self.get_channels(os.path.join(self.semantic_label_path, self.semantic_label_list[idx_another]))

        # resize
        object_image = resize(object_image, (self.img_h, self.img_w))
        object_mask = resize(object_mask, (self.img_h, self.img_w), order=0)
        layout = resize(layout, (self.img_h, self.img_w), order=0)
        background_image = resize(background_image, (self.img_h, self.img_w))
        cond_object_mask = resize(cond_object_mask, (self.img_h, self.img_w), order=0)
        cond_layout = resize(cond_layout, (self.img_h, self.img_w), order=0)

        # reshape
        object_image = np.transpose(object_image, [2, 0, 1]).astype(np.float32)
        object_mask = object_mask[None, :, :].astype(np.float32)
        background_image = np.transpose(background_image, [2, 0, 1]).astype(np.float32)
        layout = np.transpose(layout, [2, 0, 1]).astype(np.float32)
        cond_object_mask = cond_object_mask[None, :, :].astype(np.float32)
        cond_layout = np.transpose(cond_layout, [2, 0, 1]).astype(np.float32)

        # to tensor
        object_image = torch.from_numpy(object_image)
        object_mask = torch.from_numpy(object_mask)
        background_image = torch.from_numpy(background_image)
        layout = torch.from_numpy(layout)
        cond_object_mask = torch.from_numpy(cond_object_mask)
        cond_layout = torch.from_numpy(cond_layout)

        return (object_image, object_mask, background_image, layout, cond_object_mask, cond_layout, self.semantic_label_list[idx])


    def get_channels(self, layout_path):
        seg_ = Image.open(layout_path)
        seg_ = np.array(seg_)
        seg = seg_.copy()
        mask = np.zeros((seg.shape[0], seg.shape[1], self.class_num), np.uint8)

        # change 1 channel seg map to n channel binary maps
        if self.layout_flag: 
            for n in range(self.class_num):
                mask[seg_ == n, n] = 255
        else:
            color_list = self.get_color_list()
            for n in range(self.class_num):
                indices = np.where(np.all(seg_ == color_list[n], axis=-1))
                mask[indices[0], indices[1], n] = 255

        return mask
        
    def get_color_list(self):
        color_list = [[128, 64, 128],
                    [244, 35, 232],
                    [70, 70, 70],
                    [102, 102, 156],
                    [190, 153, 153],
                    [153, 153, 153],
                    [250, 170, 30],
                    [220, 220, 0],
                    [107, 142, 35],
                    [152, 251, 152],
                    [70, 130, 180],
                    [220, 20, 60],
                    [255, 0, 0],
                    [0, 0, 142],
                    [0, 0, 70],
                    [0, 60, 100],
                    [0, 80, 100],
                    [0, 0, 230],
                    [119, 11, 32]]
        return color_list
        


