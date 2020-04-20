"""Created on 2018/12/10
Usage: Inspection
Content:
    refine_dcm
    check_avphase
"""
import logging
import glob

import pydicom as dicom

from io import BytesIO


def check_dcm(dcm_filepath):
    """
    Util function to check if file is a dicom file
    the first 128 bytes are preamble
    the next 4 bytes should contain DICM otherwise it is not a dicom
    Parameters
    ----------
    dcm_filepath: cht
        The target path of dicom file
    Returns
    -------
    bool:
        check if the dicom misses information
    Reference
    ---------
    https://github.com/icometrix/dicom2nifti/blob/master/dicom2nifti/compressed_dicom.py
    """

    with open(dcm_filepath, 'rb') as fn:
        fn.seek(128)
        dcm_type = fn.read(4)
    return dcm_type == b'DICM'


def refine_dcm(dcm_filepath):
    """
    Refine DICOM file to legal format.
    Parameters
    ----------
    dcm_filepath: cht
        The target path of dicom file
    Returns
    -------
    sc: dicom object
        The dicom content from the file
    Reference
    ---------
    Refine DICOM file format https://github.com/pydicom/pydicom/issues/340
    """

    # Manually add the preamble
    byte_dcm = BytesIO()

    if not check_dcm(dcm_filepath):
        byte_dcm.write(b'\x00' * 128)
        byte_dcm.write(b'DICM')

    # Add the contents of the file
    with open(dcm_filepath, 'rb') as fn:
        byte_dcm.write(fn.read())
    byte_dcm.seek(0)

    # Read the dataset
    sc = dicom.read_file(byte_dcm)
    sc.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
    return sc


def check_avphase(dcm_filepath):
    """
    Check if the series contain A phase and V phase.
    If so, seperate and return the list for V pahse files.
    Parameters
    ----------
    dcm_filepath: cht
        The target path. All the dicom file in that path will be examinated.
    Returns
    -------
    file_list: list
        The list that contained all the file path that is in V phase
    """
    time_list = []
    file_list = []

    dcmpaths = glob.glob(dcm_filepath+'/*.dcm')

    for dcmpath in dcmpaths:
        dcmfile = refine_dcm(dcmpath)
        time_list.append(str(dcmfile[0x0008, 0x0032].value))

    if len(set(time_list)) == 2 and time_list.count(max(time_list)) > 3:
        for (filename, time) in zip(dcmpaths, time_list):
            if time == max(time_list):
                file_list.append(filename)
        check = True
    else:
        file_list = dcmpaths
        check = False
    return file_list, check