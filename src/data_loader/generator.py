import os
import glob
import ntpath

import logging

import numpy as np
import pandas as pd
import pydicom as dicom
import nrrd
import matplotlib.pyplot as plt
from datetime import datetime as ddt

from checking import refine_dcm
from preprocessing import get_pixels_hu


def get_imageposition(dcmfile):
    imageposition = np.array(list(dcmfile[0x0020, 0x0032].value))
    return imageposition


def get_pixelspacing(dcmfile):
    pixelspacing = list(dcmfile[0x0028, 0x0030].value)
    return pixelspacing


def create_box_data(tumorpath, box_save_path, sortkey, border=np.array([0, 0, 0])):
    """
    Usage: Create box data from tumorpath and add border
    """

    tumor_id = ntpath.basename(os.path.normpath(tumorpath))

    # Read label nrrd
    tumor_label, tumor_options = nrrd.read(tumorpath+'label.nrrd')
    label_shape = np.array(tumor_label.shape[1:]) \
        if len(tumor_label.shape) == 4 else np.array(tumor_label.shape)
    print(label_shape)

    # Get ordered path and find path order
    dcmpathes = sorted(glob.glob(tumorpath+'scans/*.dcm'), key=sortkey)
    dcmfile_0 = refine_dcm(dcmpathes[0])
    dcmfile_1 = refine_dcm(dcmpathes[1])

    order = 'downup' \
        if get_imageposition(dcmfile_0)[2] < \
            get_imageposition(dcmfile_1)[2] else 'updown'

    if order == 'updown':
        dcmpathes = dcmpathes[::-1]
        dcmfile_0 = refine_dcm(dcmpathes[0])
        dcmfile_1 = refine_dcm(dcmpathes[1])

    # Find each space relative origin and spacing
    img_origin = get_imageposition(dcmfile_0)
    thickness = abs(get_imageposition(dcmfile_0)[2] -
                    get_imageposition(dcmfile_1)[2])
    img_spacing = np.array(get_pixelspacing(dcmfile_0) + [float(thickness)])

    seg_origin = np.array(tumor_options['space origin'], dtype=float)
    if tumor_options['space directions'][0] == 'none':
        seg_spacing = np.diag(np.array(tumor_options['space directions'][1:])
                              .astype(float))
    else:
        seg_spacing = np.diag(np.array(tumor_options['space directions'])
                              .astype(float))

    print(seg_origin, img_origin)
    # Calculate segmetation origin index in image voxel coordinate
    seg_origin_idx = np.round((seg_origin / seg_spacing -
                               img_origin / img_spacing)).astype(int)

    # Get box origin and length
    box_origin_idx = seg_origin_idx - border
    box_length = label_shape + 2 * border

    x_orgidx, y_orgidx, z_orgidx = box_origin_idx
    x_len, y_len, z_len = box_length
    x_border, y_border, z_border = border

    base_tumor_path = box_save_path + tumor_id + '/'
    if not os.path.exists(base_tumor_path):
        os.mkdir(base_tumor_path)

    # Get DICOM scans and transfer to HU
    patient_scan = [refine_dcm(dcmpath)
                    for dcmpath in dcmpathes[z_orgidx: z_orgidx+z_len]]
    patient_hu = get_pixels_hu(patient_scan)[:, y_orgidx: y_orgidx+y_len,
                                             x_orgidx: x_orgidx+x_len]
    patient_hu = patient_hu.transpose(2, 1, 0)
    # np.save(base_tumor_path+'ctscan.npy', patient_hu)

    category_cnt = tumor_label.shape[0] \
        if tumor_options['dimension'] == 4 else 1
    category_names = [tumor_options['keyvaluepairs']
                      ['Segment{}_Name'.format(c)] for c in range(category_cnt)]

    # Get pancreas label
    for i, category_name in enumerate(category_names):
        category_label = np.zeros(box_length)
        if len(tumor_label.shape) == 4:
            category_label[x_border: x_len-x_border,
                           y_border: y_len-y_border,
                           z_border: z_len-z_border] = tumor_label[i]
        else:
            category_label[x_border: x_len-x_border,
                           y_border: y_len-y_border,
                           z_border: z_len-z_border] = tumor_label

        save_name = standard_filename(category_name)
        print(save_name)
        # np.save(base_tumor_path+'{}.npy'.format(save_name), category_label)


def standard_filename(name):
    name = name.replace('-', '')
    name = name.replace('_', '')
    name = name.replace(' ', '')
    name = name.replace('1', '')
    name = name.lower()
    return name
