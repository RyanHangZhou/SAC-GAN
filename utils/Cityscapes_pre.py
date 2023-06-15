import numpy as np
import os
import shutil
from misc import check_folder

from PIL import Image, ImagePalette
from skimage.transform import resize
from torchvision.utils import save_image

# patch_type = 'car'
# patch_type = 'bus'
# patch_type = 'person'
patch_type = 'truck'

object_image_dir = '/local-scratch2/hang/data/Cityscapes/train/panoptic'
# target_dir = '/local-scratch2/hang/detr/SAC_GAN/data/Cityscapes/train/object_mask_car'
# target_dir = '/local-scratch2/hang/detr/SAC_GAN/data/Cityscapes/train/object_mask_bus'
# target_dir = '/local-scratch2/hang/detr/SAC_GAN/data/Cityscapes/train/object_mask_person'
target_dir = '/local-scratch2/hang/detr/SAC_GAN/data/Cityscapes/train/object_mask_truck'
data_list = os.listdir(object_image_dir)
check_folder(target_dir)

for i in data_list:
    sample = os.path.join(object_image_dir, i)
    subsample_list = os.listdir(sample)
    # matching = [s for s in subsample_list if any(xs in s for xs in patch_type)]
    matching = list(filter(lambda x: patch_type in x, subsample_list))
    if len(matching)>0:
        matching.sort()
        pick_sample = matching[0]
        target_sample = os.path.join(sample, pick_sample)
        new_name = os.path.join(target_dir, os.path.basename(sample) + '.png')
        shutil.copy(target_sample, new_name)

