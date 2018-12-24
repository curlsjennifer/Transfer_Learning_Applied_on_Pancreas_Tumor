import os
import glob
import ntpath

import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, measure
import scipy.misc


def masked_2D_sampler(mask, patch_size, stride, threadshold, value=0):
    coords = []
    for z in range(mask.shape[2]):
        lbl_map = measure.label(mask[:, :, z].astype(int))
        for region in measure.regionprops(lbl_map):
            box = region.bbox
            x_len = box[2] - box[0]
            y_len = box[3] - box[1]
            for row in range(max((x_len-patch_size), 0) // stride + 1):
                for col in range(max((y_len-patch_size), 0) // stride + 1):
                    x_min = box[0]+row*stride
                    y_min = box[1]+col*stride

                    x_max = x_min+patch_size
                    if x_max > box[2]:
                        x_max = box[2]
                        x_min = x_max - patch_size
                    y_max = y_min+patch_size
                    if y_max > box[3]:
                        y_max = box[3]
                        y_min = y_max - patch_size
                    z_min = z
                    z_max = z+1
                    patch = mask[x_min:x_max, y_min:y_max, z_min:z_max]
                    if np.sum(patch)/(patch_size**2) > threadshold:
                        coords.append([value, x_min, y_min, z_min,
                                       x_max, y_max, z_max])
    return coords


def masked_3D_sampler(mask, patch_size, stride, threadshold, value=0):
    coords = []
    lbl_map = measure.label(mask.astype(int))
    for region in measure.regionprops(lbl_map):
        box = region.bbox

        xlen = box[3] - box[0]
        ylen = box[4] - box[1]
        zlen = box[5] - box[2]
        for row in range(max((xlen - patch_size), 0) // stride + 1):
            for col in range(max((ylen - patch_size), 0) // stride + 1):
                for lyr in range(max((zlen - patch_size), 0) // stride + 1):
                    x_min = box[0] + row * stride
                    y_min = box[1] + col * stride
                    z_min = box[2] + lyr * stride

                    x_max = x_min + patch_size
                    if x_max > box[3]:
                        x_max = box[3]
                        x_min = x_max - patch_size
                    y_max = y_min + patch_size
                    if y_max > box[4]:
                        y_max = box[4]
                        y_min = y_max - patch_size
                    z_max = z_min + patch_size
                    if z_max > box[5]:
                        z_max = box[5]
                        z_min = z_max - patch_size

                    patch = mask[x_min:x_max, y_min:y_max, z_min:z_max]
                    if np.sum(patch)/patch_size**3 > threadshold:
                        coords.append([value, x_min, y_min, z_min,
                                       x_max, y_max, z_max])
    return coords
