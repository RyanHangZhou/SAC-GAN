import numpy as np
import os

from PIL import Image, ImagePalette
from skimage.transform import resize
from torchvision.utils import save_image
from utils import Canny


def str2bool(x):
    return x.lower() in ('true')


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def object_extract(mask, obj, patch_s=64):
    img_h, img_w = mask.shape[1:]

    for i in range(img_h):
        line = mask[:, i, :]
        if(np.sum(line) > 0):
            h_1 = i
            break
    for i in range(img_h-1, 0, -1):
        line = mask[:, i, :]
        if(np.sum(line) > 0):
            h_2 = i
            break
    for i in range(img_w):
        column = mask[:, :, i]
        if(np.sum(column) > 0):
            w_1 = i
            break
    for i in range(img_w-1, 0, -1):
        column = mask[:, :, i]
        if(np.sum(column) > 0):
            w_2 = i
            break

    patch_mask = mask[:, h_1:h_2+1, w_1:w_2+1]
    patch_obj = obj[:, h_1:h_2+1, w_1:w_2+1]

    obj_h, obj_w = patch_mask.shape[1:]

    if(obj_h>obj_w): 
        empty_mat_left = np.zeros((obj_h, int((obj_h-obj_w)/2))).astype(np.float32)
        empty_mat_right = np.zeros((obj_h, obj_h-obj_w-int((obj_h-obj_w)/2))).astype(np.float32)
        empty_mat_left = np.expand_dims(empty_mat_left, axis=0)
        empty_mat_right = np.expand_dims(empty_mat_right, axis=0)
        empty_mat_obj_left = np.tile(empty_mat_left, (3, 1, 1))
        empty_mat_obj_right = np.tile(empty_mat_right, (3, 1, 1))
        patch_mask = np.concatenate((empty_mat_left, patch_mask, empty_mat_right), axis=2)
        patch_obj = np.concatenate((empty_mat_obj_left, patch_obj, empty_mat_obj_right), axis=2)
        obj_w = obj_h
        w_2 = w_2 + (obj_h-obj_w)
        w = img_w + (obj_h-obj_w)
    else:
        empty_mat_up = np.zeros((int((obj_w-obj_h)/2), obj_w)).astype(np.float32)
        empty_mat_down = np.zeros((obj_w-obj_h-int((obj_w-obj_h)/2), obj_w)).astype(np.float32)
        empty_mat_up = np.expand_dims(empty_mat_up, axis=0)
        empty_mat_down = np.expand_dims(empty_mat_down, axis=0)
        empty_mat_obj_up = np.tile(empty_mat_up, (3, 1, 1))
        empty_mat_obj_down = np.tile(empty_mat_down, (3, 1, 1))
        patch_mask = np.concatenate((empty_mat_up, patch_mask, empty_mat_down), axis=1)
        patch_obj = np.concatenate((empty_mat_obj_up, patch_obj, empty_mat_obj_down), axis=1)
        obj_h = obj_w
        h_1 = h_1-(obj_w-obj_h)
        h = img_h + (obj_w-obj_h)

    # image resize
    norm_mask = np.moveaxis(patch_mask, 0, -1)
    norm_obj = np.moveaxis(patch_obj, 0, -1)

    norm_mask = resize(norm_mask, (patch_s, patch_s)) # values: [0, 1]
    norm_obj = resize(norm_obj, (patch_s, patch_s)) # # values: [0, 1]

    norm_mask = np.moveaxis(norm_mask, -1, 0)
    norm_obj = np.moveaxis(norm_obj, -1, 0)

    # scaling
    s_x = patch_s*1.0/obj_w
    s_y = patch_s*1.0/obj_h

    # translation
    h_o = np.floor((h_1 + h_2)/2)
    w_o = np.floor((w_1 + w_2)/2)
    t_x = (w_o-img_w/2)/(img_w/2)
    t_y = (h_o-img_h/2)/(img_h/2)

    theta_gt = [[s_x, 0, -t_x*s_x], 
                [0, s_y, -t_y*s_y]]

    return (theta_gt, patch_mask, patch_obj, norm_mask, norm_obj)


def edge_extract(norm_obj):
    norm_obj = np.moveaxis(norm_obj*255., 0, -1)
    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    norm_obj = rgb2gray(norm_obj)
    edge_detector = Canny.CannyEdgeDetector(norm_obj, dimension=1, sigma=2)
    object_edge = edge_detector.detect_edges()
    return object_edge


def save_layout(I, path, dir_name, indx):
    I = I[0].detach().cpu().numpy()
    colormap = create_cityscapes_label_colormap()
    rendered_I = colormap[seg_to_single_channel(I, 'chw')]
    rendered_I = Image.fromarray(rendered_I.astype(dtype=np.uint8))
    rendered_I.save(os.path.join(path, dir_name+'/'+indx)) 


def save_img(I, path, dir_name, indx):
    save_image(I, os.path.join(path, dir_name+'/'+indx))


def seg_to_single_channel(seg, order='hwc'):
    if order=='chw':
        seg = np.transpose(np.squeeze(seg), [1,2,0])
    single_channel = np.argmax(seg, axis=2).astype(np.uint8)
    return single_channel


def create_cityscapes_label_colormap():
  """Creates a label colormap used in CITYSCAPES segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[0] = [128, 64, 128]
  colormap[1] = [244, 35, 232]
  colormap[2] = [70, 70, 70]
  colormap[3] = [102, 102, 156]
  colormap[4] = [190, 153, 153]
  colormap[5] = [153, 153, 153]
  colormap[6] = [250, 170, 30]
  colormap[7] = [220, 220, 0]
  colormap[8] = [107, 142, 35]
  colormap[9] = [152, 251, 152]
  colormap[10] = [70, 130, 180]
  colormap[11] = [220, 20, 60]
  colormap[12] = [255, 0, 0]
  colormap[13] = [0, 0, 142]
  colormap[14] = [0, 0, 70]
  colormap[15] = [0, 60, 100]
  colormap[16] = [0, 80, 100]
  colormap[17] = [0, 0, 230]
  colormap[18] = [119, 11, 32]
  return colormap


def colorize_mask(mask):
    palette_cityscape = [  0,  0,  0,   0,  0,  0, 111,74,   0,  81,  0, 81, 128, 64,128, 244, 35,232, 250,170,160, 230,150,140,
                          70, 70, 70, 102,102,156, 190,153,153, 180,165,180, 150,100,100, 150,120, 90, 153,153,153, 250,170, 30, 
                         220,220,  0, 107,142, 35, 152,251,152,  70,130,180, 220, 20, 60, 255,  0,  0,   0,  0,142,   0,  0, 70, 
                           0, 60,100,   0,  0, 90,   0,  0,110,   0, 80,100,   0,  0,230, 119, 11, 32] # 30 classes
    zero_pad = 256 * 3 - len(palette_cityscape)
    for i in range(zero_pad):
        palette_cityscape.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette_cityscape)

    return new_mask

    