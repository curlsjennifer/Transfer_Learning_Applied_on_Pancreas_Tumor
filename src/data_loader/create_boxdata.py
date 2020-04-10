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

from data_loader.checking import refine_dcm
from data_loader.preprocessing import get_pixels_hu, resample, find_largest


def get_imageposition(dcmfile):
    imageposition = np.array(list(dcmfile[0x0020, 0x0032].value))
    return imageposition


def get_pixelspacing(dcmfile):
    pixelspacing = list(dcmfile[0x0028, 0x0030].value)
    return pixelspacing


def get_dicominfo(tumorpath, sortkey):
    # Get ordered path and find path order
    dcmpathes = sorted(glob.glob(tumorpath + 'scans/*.dcm'), key=sortkey)
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
    thickness = abs(get_imageposition(dcmfile_0)[2]
                    - get_imageposition(dcmfile_1)[2])
    img_spacing = np.array(get_pixelspacing(dcmfile_0) + [float(thickness)])

    return dcmpathes, img_origin, thickness, img_spacing


def create_boxdata(tumorpath, sortkey,
                   border=np.array([0, 0, 0]),
                   box_save_path='',
                   fine_to_thick=False,
                   change_spacing=True,
                   new_spacing=[1, 1, 5]):
    """
    Usage: Create box data from tumorpath and add border

    Parameters
    ----------
    tumorpath (str): The path for the specific study in label data
    sortkey : Method for sorting file name
    border (numpy array): default [0, 0, 0]
    box_save_path (str): The saving path
    fine_to_thick (bool): Whether change finecut to thickcut
    chagne_spacing (bool): Resample to the new spacing
    new_spacing (numpy array): The target spacing for resample

    Returns
    -------
    Numpy array: 3D array of the box image
    List: the list containing all the label.
          In each content,
          the first value is string of label name
          the second value is the numpy array of the mask
    """

    case_id = ntpath.basename(os.path.normpath(tumorpath))

    dcmpathes, img_origin, thickness, img_spacing = get_dicominfo(
        tumorpath, sortkey)

    # Read label nrrd
    tumor_label, tumor_options = nrrd.read(tumorpath + 'label.nrrd')
    label_shape = np.array(tumor_label.shape[1:]) \
        if len(tumor_label.shape) == 4 else np.array(tumor_label.shape)

    seg_origin = np.array(tumor_options['space origin'], dtype=float)
    if np.all(np.isnan(tumor_options['space directions'][0])):
        seg_spacing = np.diag(np.array(tumor_options['space directions'][1:])
                              .astype(float))
    else:
        seg_spacing = np.diag(np.array(tumor_options['space directions'])
                              .astype(float))

    # Calculate segmetation origin index in image voxel coordinate
    seg_origin_idx = np.round((seg_origin / seg_spacing
                               - img_origin / img_spacing)).astype(int)

    # Get box origin and length
    box_origin_idx = seg_origin_idx - border
    box_length = label_shape + 2 * border

    x_orgidx, y_orgidx, z_orgidx = box_origin_idx
    x_len, y_len, z_len = box_length
    x_border, y_border, z_border = border

    # Get DICOM scans and transfer to HU
    patient_scan = [refine_dcm(dcmpath)
                    for dcmpath in dcmpathes[z_orgidx: z_orgidx + z_len]]

    patient_hu = get_pixels_hu(patient_scan)[:, y_orgidx: y_orgidx + y_len,
                                             x_orgidx: x_orgidx + x_len]
    patient_hu = patient_hu.transpose(2, 1, 0)

    if fine_to_thick:
        patient_hu = finecut_to_thickcut(patient_hu, thickness)
        spacing = [new_spacing[0], new_spacing[1], img_spacing[2]]
    else:
        spacing = new_spacing

    if change_spacing:
        patient_hu, _ = resample(patient_hu, img_spacing, spacing)

    category_cnt = tumor_label.shape[0] \
        if tumor_options['dimension'] == 4 else 1
    category_names = [tumor_options['Segment{}_Name'.format(c)]
                      for c in range(category_cnt)]

    # Get pancreas label
    label = []
    for i, category_name in enumerate(category_names):
        category_label = np.zeros(box_length)
        if len(tumor_label.shape) == 4:
            category_label[x_border: x_len - x_border,
                           y_border: y_len - y_border,
                           z_border: z_len - z_border] = tumor_label[i]
        else:
            category_label[x_border: x_len - x_border,
                           y_border: y_len - y_border,
                           z_border: z_len - z_border] = tumor_label

        if fine_to_thick:
            category_label = finecut_to_thickcut(
                category_label, thickness, label_mode=True)

        if change_spacing:
            category_label, _ = resample(category_label,
                                         img_spacing, spacing)

        save_name = standard_filename(category_name)
        label.append([save_name, category_label])

    label = manual_change(case_id, label)

    if box_save_path:
        base_tumor_path = box_save_path + case_id + '/'
        if not os.path.exists(base_tumor_path):
            os.mkdir(base_tumor_path)
        np.save(base_tumor_path + 'ctscan.npy', patient_hu)

        for item in label:
            np.save(base_tumor_path + item[0] + '.npy', item[1])

    return patient_hu, label


def create_AD_boxdata(tumorpath, sortkey,
                      border=np.array([0, 0, 0]),
                      box_save_path='',
                      fine_to_thick=False,
                      change_spacing=True,
                      new_spacing=[1, 1, 5],
                      crop_box=True):
    """
    Usage: Create box data from tumorpath and add border

    Parameters
    ----------
    tumorpath (str): The path for the specific study in label data
    sortkey : Method for sorting file name
    border (numpy array): default [0, 0, 0]
    box_save_path (str): The saving path
    fine_to_thick (bool): Whether change finecut to thickcut
    chagne_spacing (bool): Resample to the new spacing
    new_spacing (numpy array): The target spacing for resample
    crop_box (bool): whether crop to the minimal bounding box

    Returns
    -------
    Numpy array: 3D array of the box image
    List: the list containing all the label.
          In each content,
          the first value is string of label name
          the second value is the numpy array of the mask
    """

    case_id = ntpath.basename(os.path.normpath(tumorpath))

    dcmpathes, img_origin, thickness, img_spacing = get_dicominfo(
        tumorpath, sortkey)

    # Get DICOM scans and transfer to HU
    patient_scan = [refine_dcm(dcmpath) for dcmpath in dcmpathes]

    patient_hu = get_pixels_hu(patient_scan)
    patient_hu = patient_hu.transpose(2, 1, 0)

    # Read label Nifti
    label_ori_path = '/home/d/pancreas/no_label_data/result/'
    label_file = label_ori_path + case_id + '.nii/' + case_id + \
        '.nii_stage2/' + case_id + '.nii_data2/' + case_id + \
        '.nii_data2--PRED.nii.gz'
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

    if crop_box:
        box_img = patient_hu[xmin:xmax, ymin:ymax, zmin:zmax]
        box_pan = pancreas_lbl[xmin:xmax, ymin:ymax, zmin:zmax]
    else:
        box_img = patient_hu
        box_pan = pancreas_lbl

    if fine_to_thick:
        box_img = finecut_to_thickcut(box_img, thickness)
        box_pan = finecut_to_thickcut(box_pan, thickness, label_mode=True)
        spacing = [new_spacing[0], new_spacing[1], img_spacing[2]]
    else:
        spacing = new_spacing

    if change_spacing:
        box_img, _ = resample(box_img, img_spacing, spacing)
        box_pan, _ = resample(box_pan, img_spacing, spacing)

    if box_save_path:
        base_tumor_path = box_save_path + case_id + '/'
        if not os.path.exists(base_tumor_path):
            os.mkdir(base_tumor_path)
        np.save(base_tumor_path + 'ctscan.npy', box_img)
        np.save(base_tumor_path + 'pancreas.npy', box_pan)

    return box_img, box_pan


def standard_filename(name):
    """
    Usage: Change the typo and inconsistant of label's name

    Parameters
    ----------
    name (str): The string of label name

    Returns
    -------
    string: The string after correction
    """
    name = name.replace('-', '')
    name = name.replace('_', '')
    name = name.replace(' ', '')
    name = name.replace('1', '')
    name = name.lower()
    return name


def manual_change(case_id, label):
    """
    Usage: Manually change of mistake in label

    Parameters
    ----------
    case_id (str): The tumor id
    label (list) : Method for sorting file name

    Returns
    -------
    List: the list containing all the label.
    """
    if case_id in ['PT4', 'PT27', 'PT28', 'PT29', 'PT30', 'PT31']:
        for item in label:
            if item[0] == 'pancreas':
                label.append(['lesion', item[1]])

    if case_id == 'PT23':
        for item in label:
            item[0] = item[0].replace('segment2', 'lesion')

    if case_id in ['PC49', 'PC50', 'PC132', 'PC242']:
        for item in label:
            item[0] = item[0].replace('mass', 'lesion')

    return label


def finecut_to_thickcut(image, thickness, label_mode=False):
    '''
    Usage: change finecut image to thickcut

    Parameter
    ---------
    image (Numpy array): 3D numpy array image
    thickness (float): original thickness
    label_mode (bool): whether the input is label or not

    Return
    ------
    Numpy array: Image after changing to thickcut
    '''

    assert thickness < 5, "have to be less than 5!"

    zip_num = int(5 // thickness)
    remove_num = image.shape[2] % zip_num

    if not remove_num == 0:
        image = image[:, :, :-int(remove_num)]

    zip_shape = (image.shape[0], image.shape[1], int(image.shape[2] / zip_num))
    zip_img = np.zeros(zip_shape)
    for i in range(zip_shape[2]):
        zip_img[:, :, i] = np.mean(
            image[:, :, i * zip_num: (i + 1) * zip_num], axis=2)
        if label_mode:
            zip_img[zip_img < 0.3] = 0
            zip_img[zip_img >= 0.3] = 1

    return zip_img
