import numpy as np
import cv2
import os
# from skimage.transform import resize
from torchvision.utils import save_image
from PIL import Image
from PIL import ImagePalette


def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def convolve1d(I, G, axis=0):
  """
  Perform a 1D convolution for image I with mask G along a specified axis.
  :I: Input image.
  :G: Convolution mask.
  :axis: Axis for convoltion. 0 = Rows, 1 = Columns.
  """

  if (axis == 0):

    result = np.ones(I.shape, dtype=I.dtype)

    # Perform convolution in every row.
    for row in range(I.shape[0]):
      result[row, :] = np.convolve(I[row, :], G, mode='same')

    return result
    
  elif (axis == 1):

    result = np.ones(I.shape, dtype=I.dtype)

    # Perform convolution in every column.
    for col in range(I.shape[1]):
      result[:, col] = np.convolve(I[:, col], G, mode='same')

    return result

  else:
    print("Error: Invalid axis.")
    return

def save_layout(I, path, dir_name, indx):
    # palette_cityscape = [  0,  0,  0,   0,  0,  0, 111,74,   0,  81,  0, 81, 128, 64,128, 244, 35,232, 250,170,160, 230,150,140,
    #                       70, 70, 70, 102,102,156, 190,153,153, 180,165,180, 150,100,100, 150,120, 90, 153,153,153, 250,170, 30, 
    #                      220,220,  0, 107,142, 35, 152,251,152,  70,130,180, 220, 20, 60, 255,  0,  0,   0,  0,142,   0,  0, 70, 
    #                        0, 60,100,   0,  0, 90,   0,  0,110,   0, 80,100,   0,  0,230, 119, 11, 32] # 30 classes

    I = I[0].detach().cpu().numpy()
    # palette_cityscape = np.array(palette_cityscape).reshape((30, 3))
    # rendered_I = np.zeros((3, I.shape[1], I.shape[2]))
    # for i in range(I.shape[0]):
    #     temp_I = np.expand_dims(I[i, :, :], axis=0)
    #     temp_I = np.tile(temp_I, (3, 1, 1))
    #     temp_color = palette_cityscape[i]
    #     # if(i==8):
    #     #     print(temp_color)
    #     temp_color2 = np.expand_dims(temp_color, axis=[1,2])
    #     temp_color2 = np.tile(temp_color2, (1, I.shape[1], I.shape[2]))
    #     rendered_I += temp_I*temp_color2

    # rendered_I = np.moveaxis(rendered_I, 0, -1)
    # ensure_dir(os.path.join(path, dir_name+'/'))
    # cv2.imwrite(os.path.join(path, dir_name+'/2'+indx[0]), rendered_I)
   

    # rendered_I = colorize_mask(seg_to_single_channel(I, 'chw'))
    # rendered_I = Image.fromarray(seg_to_single_channel(I, 'chw'))#.convert('RGB')
    
    # save_image(rendered_I, os.path.join(path, dir_name+'/'+indx[0]))
    

    colormap = create_cityscapes_label_colormap()
    rendered_I = colormap[seg_to_single_channel(I, 'chw')]
    rendered_I = Image.fromarray(rendered_I.astype(dtype=np.uint8))
    ensure_dir(os.path.join(path, dir_name+'/'))
    rendered_I.save(os.path.join(path, dir_name+'/'+indx)) 

def save_img(I, path, dir_name, indx):
    ensure_dir(os.path.join(path, dir_name+'/'))
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
    # ignore_label = 0
    # chn_and_segID = {22: 34, ignore_label: 0, 1: 1,
    #                  2: 5, 3: 6,
    #                  4: 7, 5: 8, 6: 9, 7: 10, 8: 11, 9: 12, 10: 13,
    #                  11: 14, 12: 15, 13: 16, 14: 17,
    #                  14: 18, 15: 19, 16: 20, 17: 21, 18: 22, 19: 23, 20: 24, 21: 25, 22: 26,
    #                  23: 27, 24: 28, 25: 29, 26: 30, 27: 31, 28: 32, 29: 33}
    # mask_ = mask.copy()
    # for k, v in chn_and_segID.items():
    #     mask_[mask == k] = v

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


def str2bool(x):
    return x.lower() in ('true')

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

    