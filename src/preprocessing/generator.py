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


def create_box_data(tumorpath, border, sortkey=sortkey):
    """
    Usage: Create box data from tumorpath and add border
    """

    logging.info("Start operating " + tumorpath)

    tumor_id = ntpath.basename(os.path.normpath(tumorpath))

    # Read label nrrd
    tumor_label, tumor_options = nrrd.read(tumorpath+'label.nrrd')
    label_shape = np.array(tumor_label.shape[1:]) \
        if len(tumor_label.shape) == 4 else np.array(tumor_label.shape)

    # Get ordered path and find path order
    dcmpathes = sorted(glob.glob(tumorpath+'scans/*.dcm'), key=sortkey)
    order = 'downup' \
        if get_imageposition(dcmpathes[0])[2] < \
            get_imageposition(dcmpathes[1])[2] else 'updown'
    logging.info("{:<21}: {}".format("DICOM save order", order))

    if order == 'updown':
        dcmpathes = dcmpathes[::-1]
    assert get_imageposition(dcmpathes[0])[2] < \
        get_imageposition(dcmpathes[1])[2], "Wrong order!"

    # Find each space relative origin and spacing
    img_origin = get_imageposition(dcmpathes[0])
    thickness = abs(get_imageposition(dcmpathes[1])[2] -
                    get_imageposition(dcmpathes[0]))[2]
    img_spacing = np.array(get_pixelspacing(dcmpathes[0]) + [float(thickness)])

    seg_origin = np.array(tumor_options['space origin'], dtype=float)
    if tumor_options['space directions'][0] == 'none':
        seg_spacing = np.diag(np.array(tumor_options['space directions'][1:])
                              .astype(float))
    else:
        seg_spacing = np.diag(np.array(tumor_options['space directions'])
                              .astype(float))

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
    patient_scan = [dicom.read_file(dcmpath, force=True)
                    for dcmpath in dcmpathes[z_orgidx: z_orgidx+z_len]]
    for scan in patient_scan:
        scan.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
    patient_hu = get_pixels_hu(patient_scan)[:, y_orgidx: y_orgidx+y_len,
                                             x_orgidx: x_orgidx+x_len]
    patient_hu = patient_hu.transpose(2, 1, 0)
    np.save(base_tumor_path+'ctscan.npy', patient_hu)
    logging.info("Save CT scan numpy array.")

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

        save_name = stardard_filename(category_name)
        np.save(base_tumor_path+'{}.npy'.format(save_name), category_label)


def standard_filename(name):
    name = name.replace('-', '')
    name = name.replace('_', '')
    name = name.replace(' ', '')
    name = name.replace('1', '')
    name = name.lower()
    return name
