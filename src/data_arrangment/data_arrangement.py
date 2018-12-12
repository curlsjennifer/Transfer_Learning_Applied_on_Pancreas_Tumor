"""Created on 2018/12/06

Usage: Everying about data arrangement

Content:
    move_labeldata
    move_labeldataPC
    sort_date
    sort_series
"""

import os, glob, ntpath
import time
import logging

from shutil import copyfile, copytree, copy
from datetime import datetime as ddt

import numpy as np
import pandas as pd
import pydicom as dicom
import tqdm

from checking import refine_dcm, check_AVphase

def move_labeldata(label, brief_df, detail_df, source_scan_path, target_base_path, black_list = []):
    """
    Usage: Move DICOM and label (nrrd) to specific location.

    Parameters
    ----------
    label: cht
        The path to the label.
    brief_df: Dataframe
        The chart linking series number
    detail_df: Dataframe
        The chart linking patient number and id
    source_scan_path: cht
        The path of source DICOM
    target_base_path: cht
        Target path

    Returns
    -------
    check_copy: bool
        Whether the file have been copy or not

    """

    tumor_id = ntpath.basename(label).split('_')[0]
    dtumor_df = detail_df[detail_df['Number'] == tumor_id].reset_index()
    assert dtumor_df.shape[0] == 1, "Tumor id duplicated!"
    
    if tumor_id in black_list:
        print('Skip {} from black list!'.format(tumor_id))
        return False
    
    # Basic Info
    patient_id = dtumor_df['Code'][0]
    btumor_df = brief_df[brief_df['No.'] == tumor_id].reset_index()
    
    # TODO: Check exam date from brief and detail are same or not
    exam_date = ddt.strftime(ddt.strptime(btumor_df['Date'][0], '%Y.%m.%d'), '%Y%m%d')
    series_no = str(int(btumor_df['Series Number'][0]))
    
    # Find
    tumor_parent_path = source_scan_path + '{}/{}/'.format(patient_id, exam_date)
    target_tumor_parent_path = target_base_path + '{}/{}/'.format(patient_id, tumor_id)
    
    if not os.path.exists(target_tumor_parent_path):
        os.makedirs(target_tumor_parent_path)
    
    check_copy = False
    for dcmpath in glob.glob(tumor_parent_path+'*/*0001.dcm'):
        # Get DICOM series number and avoid dose description
        try:
            dcm_series_no = str(dicom.read_file(dcmpath)[0x0020, 0x0011].value)
        except:
            continue
        if dcm_series_no == series_no:
            # Check if A phase and V phase mix up
            file_list = check_AVphase(os.path.dirname(dcmpath))
            os.makedirs(target_tumor_parent_path + 'scans')
            for filename in file_list:
                copy(str(filename), target_tumor_parent_path + 'scans/')
            copyfile(label, target_base_path + '{}/{}/label.nrrd'.format(patient_id, tumor_id))
            check_copy = True
    if check_copy == False:
        print(tumor_id)
    return check_copy


def move_labeldata_PC(label, detail_df, source_scan_path, target_base_path, black_list = []):
    """
    Usage: Move DICOM and label (nrrd) to specific location.

    Parameters
    ----------
    label: cht
        The path to the label.
    detail_df: Dataframe
        The chart linking patient number and id
    data_type: {'normal', 'tumor', 'tumor55}
        Data type
        'normal' means normal pancreas
        'tumor' means pancreas with tumor
        'tumor55' means pancreas with tumor, and the dicom file are thick cut

    Returns
    -------
    check_copy: bool
        Whether the file have been copy or not

    """
    tumor_id = ntpath.basename(label).split('_')[0]
    dtumor_df = detail_df[detail_df['Number'] == tumor_id].reset_index()
    assert dtumor_df.shape[0] == 1, "Tumor id duplicated!"
    
    if tumor_id in black_list:
        print('Skip {} from black list!'.format(tumor_id))
        return False
    
    # Basic Info
    patient_id = dtumor_df['Code'][0]
    
    # TODO: Check exam date from brief and detail are same or not
    exam_date = str(dtumor_df['Exam Date'][0]).split('.')[0]
    
    series_no = ntpath.basename(label).split('_')[-1].split('.')[0]
    
    # Find
    tumor_parent_path = source_scan_path + '{}/{}/'.format(patient_id, exam_date)
    target_tumor_parent_path = target_base_path + '{}/{}/'.format(patient_id, tumor_id)
    
    if not os.path.exists(target_tumor_parent_path):
        os.makedirs(target_tumor_parent_path)
    
    check_copy = False
    for dcmpath in glob.glob(tumor_parent_path+'*/*I1.dcm'):
        # Get DICOM series number and avoid dose description
        try:
            dcmfile = refine_dcm(dcmpath)
            dcm_series_no = str(dcmfile[0x0020, 0x0011].value)
        except:
            continue
        if dcm_series_no == series_no:
            # Check if A phase and V phase mix up
            file_list = check_AVphase(os.path.dirname(dcmpath))
            os.makedirs(target_tumor_parent_path + 'scans')
            for filename in file_list:
                copy(str(filename), target_tumor_parent_path + 'scans/')
            copyfile(label, target_base_path + '{}/{}/label.nrrd'.format(patient_id, tumor_id))
            check_copy = True
    if check_copy == False:
        print(tumor_id)
    return check_copy


def sort_date(source_path):
    """
    Usage: Sort by study date if files are scattered in patient folder. 
        From [patient number]/xxx.dcm to [patient number]/[study date]/xxx.dcm

    Parameters
    ----------
    label: cht
        The target path.
    """

    st_tol = time.time()
    cnt = 0
    for file in tqdm(glob.glob(source_path+'/*/*.dcm')):
        try:
            date = str(dicom.read_file(file, force = True)[0x0008, 0x0020].value)
            parent_path = os.path.dirname(file) + '/' + date
            if not os.path.exists(parent_path):
                os.makedirs(parent_path)
            move(file, parent_path + '/')
            cnt += 1
        except:
            continue
    print('Done moving {} data in {} seconds'.format(cnt, time.time()-st_tol))


def sort_series(source_path):
    """
    Usage: Sort by series if files are scattered in patient folder.
        From [patient number]/[study date]/xxx.dcm 
        to [patient number]/[study date]/[series number]/xxx.dcm

    Parameters
    ----------
    label: cht
        The target path.
        
    """

    st_tol = time.time()
    cnt = 0
    error_file = []
    for case_path in tqdm(glob.glob(source_path + '/00*/*')):
        for file in glob.glob(case_path+'/*.dcm'):
            try:
                series_number = str(dicom.read_file(file, force = True)[0x0020, 0x0011].value)
                parent_path = os.path.dirname(file) + '/' + series_number
                if not os.path.exists(parent_path):
                    os.makedirs(parent_path)
                move(file, parent_path + '/')
                cnt +=1 
            except:
                error_file.append(file)
                continue
    print('Done moving {} data in {} seconds'.format(cnt, time.time()-st_tol))