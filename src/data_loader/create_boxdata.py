"""Created on 2019/01/13

Usage: From label data to box data

Content:
    get_imageposition
    get_pixelspacing
    create_boxdata
    create_AD_boxdata
"""

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
from skimage import morphology, measure
import nibabel as nib
import scipy

from checking import refine_dcm
from preprocessing import get_pixels_hu, resample, find_largest


def get_imageposition(dcmfile):
    imageposition = np.array(list(dcmfile[0x0020, 0x0032].value))
    return imageposition


def get_pixelspacing(dcmfile):
    pixelspacing = list(dcmfile[0x0028, 0x0030].value)
    return pixelspacing


def create_boxdata(tumorpath, sortkey,
                   border=np.array([0, 0, 0]),
                   box_save_path='',
                   change_spacing=True,
                   new_spacing=[1, 1, 5]):
    """
    Usage: Create box data from tumorpath and add border

    Parameters
    ----------
    tumorpath (str): The path for the specific study in label data
    sortkey : Method for sorting file name
    border (numpy array): default [0, 0, 0]
    savefile (bool): Whether saving box data or not
    box_save_path (str): The saving path
    new_spacing (numpy array): The target spacing for resample

    Returns
    -------
    Numpy array: 3D array of the box image
    List: the list containing all the label.
          In each content,
          the first value is string of label name
          the second value is the numpy array of the mask
    """

    tumor_id = ntpath.basename(os.path.normpath(tumorpath))

    # Read label nrrd
    tumor_label, tumor_options = nrrd.read(tumorpath+'label.nrrd')
    label_shape = np.array(tumor_label.shape[1:]) \
        if len(tumor_label.shape) == 4 else np.array(tumor_label.shape)

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
    if np.all(np.isnan(tumor_options['space directions'][0])):
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

    # Get DICOM scans and transfer to HU
    patient_scan = [refine_dcm(dcmpath)
                    for dcmpath in dcmpathes[z_orgidx: z_orgidx+z_len]]

    patient_hu = get_pixels_hu(patient_scan)[:, y_orgidx: y_orgidx+y_len,
                                             x_orgidx: x_orgidx+x_len]
    patient_hu = patient_hu.transpose(2, 1, 0)

    if change_spacing:
        patient_hu, _ = resample(patient_hu, img_spacing, new_spacing)

    category_cnt = tumor_label.shape[0] \
        if tumor_options['dimension'] == 4 else 1
    category_names = [tumor_options['Segment{}_Name'.format(c)] for c in range(category_cnt)]

    # Get pancreas label
    label = []
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

        if change_spacing:
            category_label, _ = resample(category_label,
                                         img_spacing, new_spacing)
        save_name = standard_filename(category_name)
        label.append([save_name, category_label])

    if box_save_path:
        base_tumor_path = box_save_path + tumor_id + '/'
        if not os.path.exists(base_tumor_path):
            os.mkdir(base_tumor_path)
        np.save(base_tumor_path + 'ctscan.npy', patient_hu)

        for item in label:
            np.save(base_tumor_path + item[0] + '.npy', item[1])

    return patient_hu, label


def create_AD_boxdata(tumorpath, sortkey,
                      border=np.array([0, 0, 0]),
                      box_save_path='',
                      change_spacing=True,
                      new_spacing=[1, 1, 5]):
    """
    Usage: Create box data from tumorpath and add border

    Parameters
    ----------
    tumorpath (str): The path for the specific study in label data
    sortkey : Method for sorting file name
    border (numpy array): default [0, 0, 0]
    savefile (bool): Whether saving box data or not
    box_save_path (str): The saving path
    new_spacing (numpy array): The target spacing for resample

    Returns
    -------
    Numpy array: 3D array of the box image
    List: the list containing all the label.
          In each content,
          the first value is string of label name
          the second value is the numpy array of the mask
    """

    ID = ntpath.basename(os.path.normpath(tumorpath))

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

    # Get DICOM scans and transfer to HU
    patient_scan = [refine_dcm(dcmpath) for dcmpath in dcmpathes]

    patient_hu = get_pixels_hu(patient_scan)
    patient_hu = patient_hu.transpose(2, 1, 0)

    # Read label Nifti
    label_ori_path = '/home/d/pancreas/no_label_data/result/'
    label_file = label_ori_path + ID + '.nii/' + ID + '.nii_stage2/' \
        + ID + '.nii_data2/' + ID + '.nii_data2--PRED.nii.gz'
    label = nib.load(label_file)
    lbl = label.get_data()
    lbl_append = scipy.ndimage.zoom(lbl, 2, order=0)
    lbl_append = lbl_append[:, :, :patient_hu.shape[2]]
    lbl_append = lbl_append[::-1, ::-1, :]

    pancreas_lbl = np.zeros(lbl_append.shape)
    pancreas_lbl[np.where(lbl_append == 7)] = 1
    pancreas_lbl = find_largest(pancreas_lbl)

    index = np.where(pancreas_lbl == 1)
    xmin, ymin, zmin = np.min(index, axis=1)
    xmax, ymax, zmax = np.max(index, axis=1)

    # box_img = patient_hu[xmin:xmax, ymin:ymax, zmin:zmax]
    box_img = patient_hu

    # box_pan = pancreas_lbl[xmin:xmax, ymin:ymax, zmin:zmax]
    box_pan = pancreas_lbl

    if change_spacing:
        box_img, _ = resample(box_img, img_spacing, new_spacing)
        box_pan, _ = resample(box_pan, img_spacing, new_spacing)
    box_pan = morphology.erosion(box_pan, np.ones([3, 3, 3]))

    if box_save_path:
        base_tumor_path = box_save_path + ID + '/'
        if not os.path.exists(base_tumor_path):
            os.mkdir(base_tumor_path)
        np.save(base_tumor_path + 'ctscan.npy', box_img)
        np.save(base_tumor_path + 'pancreas.npy', box_pan)

    return box_img, box_pan


def standard_filename(name):
    name = name.replace('-', '')
    name = name.replace('_', '')
    name = name.replace(' ', '')
    name = name.replace('1', '')
    name = name.lower()
    return name