import numpy as np
import cv2
from skimage.transform import resize
from torchvision.utils import save_image


def object_extract(mask, obj, flag, indx):
    N, _, h, w = np.shape(mask)
    for n in range(N): 
        for i in range(h):
            line = mask[:, :, i, :]
            if(np.sum(line) > 0):
                h_1 = i
                break
        for i in range(h-1, 0, -1):
            line = mask[:, :, i, :]
            if(np.sum(line) > 0):
                h_2 = i
                break
        for i in range(w):
            column = mask[:, :, :, i]
            if(np.sum(column) > 0):
                w_1 = i
                break
        for i in range(w-1, 0, -1):
            column = mask[:, :, :, i]
            if(np.sum(column) > 0):
                w_2 = i
                break
        new_mask = mask[n, :, h_1:h_2+1, w_1:w_2+1]
        new_obj = obj[n, :, h_1:h_2+1, w_1:w_2+1]
        _, obj_h, obj_w = np.shape(new_mask)

        if(flag==1):
            if(obj_h>obj_w): 
                empty_mat_left = np.zeros((obj_h, int((obj_h-obj_w)/2)))
                empty_mat_right = np.zeros((obj_h, obj_h-obj_w-int((obj_h-obj_w)/2)))
                empty_mat_left = np.expand_dims(empty_mat_left, axis=0)
                empty_mat_right = np.expand_dims(empty_mat_right, axis=0)
                empty_mat_obj_left = np.tile(empty_mat_left, (3, 1, 1))
                empty_mat_obj_right = np.tile(empty_mat_right, (3, 1, 1))
                new_mask = np.concatenate((empty_mat_left, new_mask, empty_mat_right), axis=2)
                new_obj = np.concatenate((empty_mat_obj_left, new_obj, empty_mat_obj_right), axis=2)
            else:
                empty_mat_up = np.zeros((int((obj_w-obj_h)/2), obj_w))
                empty_mat_down = np.zeros((obj_w-obj_h-int((obj_w-obj_h)/2), obj_w))
                empty_mat_up = np.expand_dims(empty_mat_up, axis=0)
                empty_mat_down = np.expand_dims(empty_mat_down, axis=0)
                empty_mat_obj_up = np.tile(empty_mat_up, (3, 1, 1))
                empty_mat_obj_down = np.tile(empty_mat_down, (3, 1, 1))
                new_mask = np.concatenate((empty_mat_up, new_mask, empty_mat_down), axis=1)
                new_obj = np.concatenate((empty_mat_obj_up, new_obj, empty_mat_obj_down), axis=1)

    return (np.float32(new_mask), np.float32(new_obj))


def batch_object_extract(mask, obj, new_h, new_w, indx, flag):
    N, _, h, w = np.shape(mask)
    batch_s_x = np.zeros(N)
    batch_s_y = np.zeros(N)
    batch_t_x = np.zeros(N)
    batch_t_y = np.zeros(N)
    for n in range(N): 
        for i in range(h):
            line = mask[n, :, i, :]
            if(np.sum(line) > 0):
                h_1 = i
                break
        for i in range(h-1, 0, -1):
            line = mask[n, :, i, :]
            if(np.sum(line) > 0):
                h_2 = i
                break
        for i in range(w):
            column = mask[n, :, :, i]
            if(np.sum(column) > 0):
                w_1 = i
                break
        for i in range(w-1, 0, -1):
            column = mask[n, :, :, i]
            if(np.sum(column) > 0):
                w_2 = i
                break
        new_mask = mask[n, :, h_1:h_2+1, w_1:w_2+1]
        new_obj = obj[n, :, h_1:h_2+1, w_1:w_2+1]
        _, obj_h, obj_w = np.shape(new_mask)

        if(flag==1):
            if(obj_h>obj_w): 
                empty_mat_left = np.zeros((obj_h, int((obj_h-obj_w)/2)))
                empty_mat_right = np.zeros((obj_h, obj_h-obj_w-int((obj_h-obj_w)/2)))
                empty_mat_left = np.expand_dims(empty_mat_left, axis=0)
                empty_mat_right = np.expand_dims(empty_mat_right, axis=0)
                empty_mat_obj_left = np.tile(empty_mat_left, (3, 1, 1))
                empty_mat_obj_right = np.tile(empty_mat_right, (3, 1, 1))
                new_mask = np.concatenate((empty_mat_left, new_mask, empty_mat_right), axis=2)
                new_obj = np.concatenate((empty_mat_obj_left, new_obj, empty_mat_obj_right), axis=2)
                obj_w = obj_h
                w_2 = w_2 + (obj_h-obj_w)
                w = w + (obj_h-obj_w)
            else:
                empty_mat_up = np.zeros((int((obj_w-obj_h)/2), obj_w))
                empty_mat_down = np.zeros((obj_w-obj_h-int((obj_w-obj_h)/2), obj_w))
                empty_mat_up = np.expand_dims(empty_mat_up, axis=0)
                empty_mat_down = np.expand_dims(empty_mat_down, axis=0)
                empty_mat_obj_up = np.tile(empty_mat_up, (3, 1, 1))
                empty_mat_obj_down = np.tile(empty_mat_down, (3, 1, 1))
                new_mask = np.concatenate((empty_mat_up, new_mask, empty_mat_down), axis=1)
                new_obj = np.concatenate((empty_mat_obj_up, new_obj, empty_mat_obj_down), axis=1)
                obj_h = obj_w
                h_1 = h_1-(obj_w-obj_h)
                h = h + (obj_w-obj_h)

        # image resize
        new_mask = np.moveaxis(new_mask, 0, -1)
        new_obj = np.moveaxis(new_obj, 0, -1)
        # cv2.imwrite('results/obj_mask'+indx[n], new_mask*255)
        # cv2.imwrite('results/oobj'+indx[n], new_obj*255)

        new_mask = resize(new_mask, (new_h, new_w)) # values: [0, 1]
        new_obj = resize(new_obj, (new_h, new_w)) # # values: [0, 1]

        # cv2.imwrite('results/res_obj_mask'+indx[n], new_mask*255)
        # cv2.imwrite('results/res_oobj'+indx[n], new_obj*255)
        
        new_mask = np.moveaxis(new_mask, -1, 0)
        new_obj = np.moveaxis(new_obj, -1, 0)

        # scaling parameters
        batch_s_x[n] = new_w*1.0/obj_w
        batch_s_y[n] = new_h*1.0/obj_h

        # translation parameters
        h_o = np.floor((h_1 + h_2)/2)
        w_o = np.floor((w_1 + w_2)/2)
        batch_t_x[n] = (w_o-w/2)/(w/2)
        batch_t_y[n] = (h_o-h/2)/(h/2)

        if(n==0):
            batch_mask = np.expand_dims(new_mask, axis=0)
            batch_obj = np.expand_dims(new_obj, axis=0)
        else:
            tmp_mask = np.expand_dims(new_mask, axis=0)
            tmp_obj = np.expand_dims(new_obj, axis=0)
            batch_mask = np.concatenate((batch_mask, tmp_mask), axis=0)
            batch_obj = np.concatenate((batch_obj, tmp_obj), axis=0)

    theta_gt = [[batch_s_x, np.zeros(N), -batch_t_x*batch_s_x], 
                [np.zeros(N), batch_s_y, -batch_t_y*batch_s_y]]
    theta_gt = np.moveaxis(theta_gt, -1, 0) # (4, 2, 3)

    return (np.float32(batch_mask), np.float32(batch_obj), np.float32(theta_gt))




