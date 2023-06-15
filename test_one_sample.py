import imageio
import numpy as np
import os
import shutil
import torch

from glob import glob
from main import parse_args, check_args
from model import SAC_GAN
from PIL import Image
from skimage.transform import resize
from utils import misc


img_h = 540
img_w = 960
class_num = 19
patch_s = 64


def read_image(image_path):
    image = imageio.imread(image_path)
    if len(image.shape) == 3:
        image = resize(image, (img_h, img_w)) # resize: automatically convert scale 255->1
        image = np.transpose(image, [2, 0, 1]).astype(np.float32)[0:3, :, :]
    else:
        image = resize(image, (img_h, img_w), order=0)
        image = image[None, :, :].astype(np.float32)
    return image


def get_color_list():
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


def get_channels(layout_path, is_layout_real=True):
    seg_ = Image.open(layout_path)
    seg_ = np.array(seg_)
    seg = seg_.copy()
    mask = np.zeros((seg.shape[0], seg.shape[1], class_num), np.uint8)
    # change 1 channel seg map to n channel binary maps
    if is_layout_real: 
        for n in range(class_num):
            mask[seg_ == n, n] = 255
    else:
        seg_ = seg_[:, :, 0:3]
        color_list = get_color_list()
        for n in range(class_num):
            indices = np.where(np.all(seg_ == color_list[n], axis=-1))
            mask[indices[0], indices[1], n] = 255
    mask = resize(mask, (img_h, img_w), order=0)
    mask = np.transpose(mask, [2, 0, 1]).astype(np.float32)
    return mask


def load(dir, step):
    print('Load model: ', os.path.join(dir, 'params_%07d.pt' % step))
    params = torch.load(os.path.join(dir, 'params_%07d.pt' % step))
    sacgan.load_state_dict(params['sacgan'])


def test(result_dir):
    model_list = glob(os.path.join(result_dir, 'ckpt', '*.pt'))
    if not len(model_list) == 0:
        model_list.sort()
        iter = int(model_list[-1].split('_')[-1].split('.')[0])
        load(os.path.join(result_dir, 'ckpt'), iter)
        print(" [*] Load SUCCESS")
    else:
        print(" [*] Load FAILURE")
        return

    with torch.no_grad():
        self.sacgan.eval()
        for target in self.test_dataloader:
            self.sacgan.inference(target, self.result_dir)


im_root = '/local-scratch2/hang/DINO/data_Cityscapes_full_inpaint/val2017'
object_image_root = '/local-scratch2/hang/data/Cityscapes/test/original_image'
object_mask_root = '/local-scratch2/hang/data/Cityscapes/test/panoptic'
semantic_layout_root = '/local-scratch2/hang/data/Cityscapes/test/semantic_label'
patch_root = im_root + '_patch_clean'
results_root = 'SACGAN_sequence'
misc.check_folder(results_root)

for filename in os.listdir(patch_root):
    print(filename)
    
    background_image = read_image(os.path.join(im_root, filename + '.png'))
    patch_list = os.listdir(os.path.join(patch_root, filename))
    object_mask_dir = os.path.join(object_mask_root, filename)
    object_image_dir = os.path.join(object_image_root, filename)
    object_image = read_image(object_image_dir  + '.png')
    layout_dir = os.path.join(semantic_layout_root, filename)
    layout = get_channels(os.path.join(semantic_layout_root, layout_dir + '_prediction.png'), True)

    for sample in patch_list:

        """ 1. Load data """
        object_mask_path = os.path.join(object_mask_dir, sample)
        object_mask = read_image(object_mask_path)[0, :, :]
        object_mask = np.expand_dims(object_mask, 0)

        _, patch_mask, patch_obj, norm_mask, norm_obj = misc.object_extract(object_mask, object_image, patch_s=patch_s)
        
        object_edge = misc.edge_extract(norm_obj)
        object_edge = np.expand_dims(object_edge, 0)
        object_edge_ = torch.from_numpy(object_edge/255.).float()[None, :]

        background_image_ = torch.from_numpy(background_image)[None, :]
        layout_ = torch.from_numpy(layout)[None, :]
        object_mask_ = torch.from_numpy(object_mask)[None, :]
        patch_mask_ = torch.from_numpy(np.float32(patch_mask))[None, :]
        patch_obj_ = torch.from_numpy(np.float32(patch_obj))[None, :]
        norm_mask_ = torch.from_numpy(np.float32(norm_mask))[None, :]
        norm_obj_ = torch.from_numpy(np.float32(norm_obj))[None, :]

        target = {}
        target["background_image"] = background_image_
        target["layout"] = layout_
        target["object_mask"] = object_mask_
        target["patch_obj"] = patch_obj_
        target["patch_mask"] = patch_mask_
        target["norm_mask"] = norm_mask_
        target["norm_obj"] = norm_obj_
        target["object_edge"] = object_edge_
        target["sample"] = [filename + '_' + sample]

        """ 2. Load model """
        object_type = sample[:-7]
        if object_type == 'car':
            target_label = 13
        # elif object_type == 'truck':
        #     target_label = 14
        elif object_type == 'bus':
            target_label = 15
        elif object_type == 'person':
            target_label = 11
        else:
            continue
                
        args = parse_args()
        args.target_class = object_type
        args.result_dir = results_root
        args.batch_size = 1
        check_args(args)
        engine = SAC_GAN(args)
        engine.single_build_model(args)

        """ 3. Inference """
        engine.single_test(object_type, target, results_root)

        """4. Iterate loading """
        background_path = os.path.join(results_root, 'test/image', filename + '_' + sample)
        background_image = read_image(background_path)
        layout_dir = os.path.join(results_root, 'test/layout', filename + '_' + sample)
        layout = get_channels(layout_dir, False)

        new_img_dir = os.path.join(results_root, 'test/image_')
        misc.check_folder(new_img_dir)
        shutil.copy(background_path, os.path.join(new_img_dir, filename + '.png'))
        new_layout_dir = os.path.join(results_root, 'test/layout_')
        misc.check_folder(new_layout_dir)
        shutil.copy(layout_dir, os.path.join(new_layout_dir, filename + '.png'))
