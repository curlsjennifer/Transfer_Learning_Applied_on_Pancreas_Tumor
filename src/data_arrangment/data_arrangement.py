"""Created on 2018/12/06

Usage: Everying about data arrangement

Content:
    move_labeldata
    move_labeldataPC
"""

import os, glob, ntpath
import time
import logging

from shutil import copyfile, copytree, copy
from datetime import datetime as ddt

import numpy as np
import pandas as pd
import pydicom as dicom

def move_labeldata(label, brief_descrip_path, data_type):
    """
    Usage: Move DICOM and label (nrrd) to specific location.

    Parameters
    ----------
    label: cht
        The path to the label.
    brief_descrip_path: cht
        The path to the description file.
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

    brief_df = pd.read_excel(brief_descrip_path).fillna('')

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
    tumor_parent_path = source_scan_path + '{}/{}/{}/'.format(data_type, patient_id, exam_date)
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
            time_list = []
            dcmfiles = glob.glob(os.path.dirname(dcmpath)+'/*.dcm')
            for dcmfile in dcmfiles:
                time_list.append(str(dicom.read_file(dcmfile)[0x0008, 0x0032].value))

            if len(set(time_list)) == 2:
                os.makedirs(target_tumor_parent_path + 'scans')
                for (file, time) in zip(dcmfiles, time_list):
                    if time == max(time_list):
                        copy(file, target_tumor_parent_path + 'scans/')
                copyfile(label, target_base_path + '{}/{}/label.nrrd'.format(patient_id, tumor_id))
                check_copy = True
                continue
            else:
                copytree(os.path.dirname(dcmpath), target_tumor_parent_path + 'scans/')
                copyfile(label, target_base_path + '{}/{}/label.nrrd'.format(patient_id, tumor_id))
                check_copy = True
                continue
    if check_copy == False:
        print(tumor_id)
    return check_copy

def move_labeldata_PC(label, data_type):
    '''
    Usage: Move DICOM and label (nrrd) to specific location.
    '''
    tumor_id = ntpath.basename(label).split('_')[0]
    dtumor_df = detail_df[detail_df['Number'] == tumor_id].reset_index()
    assert dtumor_df.shape[0] == 1, "Tumor id duplicated!"
    
    if tumor_id in black_list:
        print('Skip {} from black list!'.format(tumor_id))
        return False
    
    # Basic Info
    patient_id = dtumor_df['Code'][0]
    
    # TODO: Check exam date from brief and detail are same or not
    exam_date = dtumor_df['Exam Date'][0]
    
    series_no = ntpath.basename(label).split('_')[-1].split('.')[0]
    
    # Find
    tumor_parent_path = source_scan_path + '{}/{}/{}/'.format(data_type, patient_id, exam_date)
    target_tumor_parent_path = target_base_path + '{}/{}/'.format(patient_id, tumor_id)
    
    if not os.path.exists(target_tumor_parent_path):
        os.makedirs(target_tumor_parent_path)
    
    check_copy = False
    for dcmpath in glob.glob(tumor_parent_path+'*/*I1.dcm'):
        # Get DICOM series number and avoid dose description
        try:
            dcm_series_no = str(dicom.read_file(dcmpath, force = True)[0x0020, 0x0011].value)
        except:
            continue
        if dcm_series_no == series_no:
            # Check if A phase and V phase mix up
            time_list = []
            dcmfiles = glob.glob(os.path.dirname(dcmpath)+'/*.dcm')
            for dcmfile in dcmfiles:
                time_list.append(str(dicom.read_file(dcmfile, force = True)[0x0008, 0x0032].value))
            if len(set(time_list)) == 2:
                print("Find different phase in", tumor_id)
                os.makedirs(target_tumor_parent_path + 'scans')
                for (file, time) in zip(dcmfiles, time_list):
                    if time == max(time_list):
                        copy(file, target_tumor_parent_path + 'scans/')
                copyfile(label, target_base_path + '{}/{}/label.nrrd'.format(patient_id, tumor_id))
                check_copy = True
                continue
            else:
                copytree(os.path.dirname(dcmpath), target_tumor_parent_path + 'scans/')
                copyfile(label, target_base_path + '{}/{}/label.nrrd'.format(patient_id, tumor_id))
                check_copy = True
                continue
    if check_copy == False:
        print(tumor_id)
    return check_copy

def sort_date(source_path):
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