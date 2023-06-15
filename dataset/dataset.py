
import imageio
import numpy as np
import os
import random
import torch

from glob import glob
from PIL import Image
from skimage.transform import resize
from utils import misc


class CityscapesDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        dataset_path, 
        img_h = 540, 
        img_w = 960, 
        class_num = 35, 
        patch_s = 64,
        is_layout_real = True, 
        target_class = 'car',
        is_train = True,
        is_random = True,
        ):

        self.dataset_path = dataset_path
        self.img_h = img_h
        self.img_w = img_w
        self.class_num = class_num
        self.patch_s = patch_s
        self.is_layout_real = is_layout_real
        self.target_class = target_class
        self.is_train = is_train
        self.is_random = is_random

        self.suffix = 'train' if self.is_train == True else 'test'

        self.object_image_path = os.path.join(self.dataset_path, self.suffix, 'object_image')
        if self.is_train: 
            # self.object_mask_path = os.path.join(self.dataset_path, self.suffix, 'object_mask_' + self.target_class)
            self.object_mask_path = os.path.join(self.dataset_path, self.suffix, 'object_mask')
        else:
            self.object_mask_path = os.path.join(self.dataset_path, self.suffix, 'object_mask')
        self.background_image_path = os.path.join(self.dataset_path, self.suffix, 'background_image')
        self.semantic_label_path = os.path.join(self.dataset_path, self.suffix, 'semantic_label')

        self.sample_list = [x.split('/')[-1] for x in glob(self.object_mask_path + '/*')]

        if self.is_random:
            self.sample_index = random.sample(range(len(self.sample_list)), k=len(self.sample_list))
        else:
            self.sample_index = range(len(self.sample_list))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        if self.is_train == True: 
            rand_idx = self.sample_index[int(idx)]
            sample = self.sample_list[rand_idx]
            rand_idx_ = np.mod(rand_idx+1, len(self.sample_list))
            sample_ = self.sample_list[rand_idx_]

            object_image = self.read_image(os.path.join(self.object_image_path, sample))
            object_mask = self.read_image(os.path.join(self.object_mask_path, sample))
            background_image = self.read_image(os.path.join(self.background_image_path, sample))
            layout = self.get_channels(os.path.join(self.semantic_label_path, sample))
            cond_object_mask = self.read_image(os.path.join(self.object_mask_path, sample_))
            cond_layout = self.get_channels(os.path.join(self.semantic_label_path, sample_))

            theta_gt, patch_mask, patch_obj, norm_mask, norm_obj = misc.object_extract(object_mask, object_image, patch_s=self.patch_s)
            theta_gt_ref = misc.object_extract(cond_object_mask, object_image, patch_s=self.patch_s)[0]

            object_edge = misc.edge_extract(norm_obj)
            object_edge = np.expand_dims(object_edge, 0)
            object_edge = torch.from_numpy(object_edge/255.).float()

            theta_gt = torch.from_numpy(np.float32(theta_gt))
            theta_gt_ref = torch.from_numpy(np.float32(theta_gt_ref))
            background_image = torch.from_numpy(background_image)
            layout = torch.from_numpy(layout)
            cond_layout = torch.from_numpy(cond_layout)
            object_mask = torch.from_numpy(object_mask)
            patch_mask = torch.from_numpy(np.float32(patch_mask))
            patch_obj = torch.from_numpy(np.float32(patch_obj))
            norm_mask = torch.from_numpy(np.float32(norm_mask))
            norm_obj = torch.from_numpy(np.float32(norm_obj))

            target = {}
            target["theta_gt"] = theta_gt
            target["theta_gt_ref"] = theta_gt_ref
            target["background_image"] = background_image
            target["layout"] = layout
            target["cond_layout"] = cond_layout
            target["object_mask"] = object_mask
            target["norm_mask"] = norm_mask
            target["norm_obj"] = norm_obj
            target["object_edge"] = object_edge
            target["sample"] = sample

        else:
            rand_idx = self.sample_index[int(idx)]
            sample = self.sample_list[rand_idx]

            object_image = self.read_image(os.path.join(self.object_image_path, sample))
            object_mask = self.read_image(os.path.join(self.object_mask_path, sample))
            background_image = self.read_image(os.path.join(self.background_image_path, sample))
            layout = self.get_channels(os.path.join(self.semantic_label_path, sample))

            _, patch_mask, patch_obj, norm_mask, norm_obj = misc.object_extract(object_mask, object_image, patch_s=self.patch_s)
            object_edge = misc.edge_extract(norm_obj)
            object_edge = np.expand_dims(object_edge, 0)
            object_edge = torch.from_numpy(object_edge/255.).float()

            background_image = torch.from_numpy(background_image)
            layout = torch.from_numpy(layout)
            object_mask = torch.from_numpy(object_mask)
            patch_mask = torch.from_numpy(np.float32(patch_mask))
            patch_obj = torch.from_numpy(np.float32(patch_obj))
            norm_mask = torch.from_numpy(np.float32(norm_mask))
            norm_obj = torch.from_numpy(np.float32(norm_obj))

            target = {}
            target["background_image"] = background_image
            target["layout"] = layout
            target["object_mask"] = object_mask
            target["patch_obj"] = patch_obj
            target["patch_mask"] = patch_mask
            target["norm_mask"] = norm_mask
            target["norm_obj"] = norm_obj
            target["object_edge"] = object_edge
            target["sample"] = sample

        return target

    def read_image(self, image_path):
        image = imageio.imread(image_path)

        if len(image.shape) == 3:
            image = resize(image, (self.img_h, self.img_w)) # resize: automatically convert scale 255->1
            image = np.transpose(image, [2, 0, 1]).astype(np.float32)[0:3, :, :]
        else:
            image = resize(image, (self.img_h, self.img_w), order=0)
            image = image[None, :, :].astype(np.float32)

        return image

    def get_channels(self, layout_path):
        seg_ = Image.open(layout_path)
        seg_ = np.array(seg_)
        seg = seg_.copy()
        mask = np.zeros((seg.shape[0], seg.shape[1], self.class_num), np.uint8)

        # change 1 channel seg map to n channel binary maps
        if self.is_layout_real: 
            for n in range(self.class_num):
                mask[seg_ == n, n] = 255
        else:
            seg_ = seg_[:, :, 0:3]
            color_list = self.get_color_list()
            for n in range(self.class_num):
                indices = np.where(np.all(seg_ == color_list[n], axis=-1))
                mask[indices[0], indices[1], n] = 255

        mask = resize(mask, (self.img_h, self.img_w), order=0)
        mask = np.transpose(mask, [2, 0, 1]).astype(np.float32)

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
        


class ChairDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, img_h, img_w, class_num, is_layout_real, is_train=True):

        self.img_h = img_h
        self.img_w = img_w
        self.class_num = class_num
        self.is_layout_real = is_layout_real

        self.suffix = 'train' if is_train == True else 'test'

        self.object_image_path = os.path.join(dataset, self.suffix, 'object_image')
        self.object_mask_path = os.path.join(dataset, self.suffix, 'object_mask')
        self.background_image_path = os.path.join(dataset, self.suffix, 'background_image')
        self.semantic_label_path = os.path.join(dataset, self.suffix, 'semantic_label')

        self.sample_list = [x.split('/')[-1] for x in glob(self.object_mask_path + '/*')]

        if is_random:
            self.sample_index = random.sample(range(len(self.sample_list)), k=len(self.sample_list))
        else:
            self.sample_index = range(len(self.sample_list))

    def __len__(self):
        return len(self.object_mask_list)

    def __getitem__(self, idx):
        idx = int(idx)
        idx = self.sample_index[idx]

        object_image = imageio.imread(os.path.join(self.object_image_path, self.object_mask_list[idx][:-23]+'_leftImg8bit.png'))
        object_mask = imageio.imread(os.path.join(self.object_mask_path, self.object_mask_list[idx]))
        background_image = imageio.imread(os.path.join(self.background_image_path, self.object_mask_list[idx][:-23]+'_leftImg8bit.png'))
        layout = self.get_channels(os.path.join(self.semantic_label_path, self.object_mask_list[idx][:-23]+'_leftImg8bit_prediction.png'))

        idx_another = np.mod(idx+1, len(self.object_mask_list))
        cond_object_mask = imageio.imread(os.path.join(self.object_mask_path, self.object_mask_list[idx_another]))
        cond_layout = self.get_channels(os.path.join(self.semantic_label_path, self.object_mask_list[idx_another][:-23]+'_leftImg8bit_prediction.png'))

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

        return (object_image, object_mask, background_image, layout, cond_object_mask, cond_layout, self.object_mask_list[idx])


    def get_channels(self, layout_path):
        seg_ = Image.open(layout_path)
        seg_ = np.array(seg_)
        seg = seg_.copy()
        mask = np.zeros((seg.shape[0], seg.shape[1], self.class_num), np.uint8)

        # change 1 channel seg map to n channel binary maps
        if self.is_layout_real: 
            for n in range(self.class_num):
                mask[seg_ == n, n] = 255
        else:
            seg_ = seg_[:, :, 0:3]
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
        




