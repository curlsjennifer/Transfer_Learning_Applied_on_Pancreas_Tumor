import logging

import pydicom as dicom

from io import BytesIO

def check_dcm(dcm_filepath):
    """
    Util function to check if file is a dicom file
    the first 128 bytes are preamble
    the next 4 bytes should contain DICM otherwise it is not a dicom
    
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