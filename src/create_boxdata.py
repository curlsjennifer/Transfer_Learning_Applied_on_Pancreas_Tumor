"""
filename : create_boxdata.py
author : WanYun, Yang
date : 2020/04/11
description :
    create box data for ntuh, msd, and tcia dataset.
"""

import os
import glob
import numpy as np
import pandas as pd

from utils import load_config
from data_loader.data_loader import save_boxdata

# Record data path
config = load_config('./configs/basic.yml')
list_path = '/data2/pancreas/Nifti_data/data_list.csv'
result_path = '/data2/pancreas/box_data/wanyun/patient_list.npy'
tcia_path = '/data2/open_dataset/pancreas/Pancreas-CT/'
msd_path = '/data2/open_dataset/MSD/Task07_Pancreas/imagesTr/'

# Create file list for ntuh data
df = pd.read_csv(list_path, converters={'add_date': str})
nh_list = list(df[(df['type'] == 'healthy') &
                  (df['diff_patient_list'])]['case_id'])
nc_list = list(df[(df['type'] == 'tumor') &
                  (df['diff_patient_list'])]['case_id'])

eh_list = [files.split('/')[-1] for files in glob.glob(tcia_path + 'P*')]
ec_list = [files.split('/')[-1] for files in glob.glob(msd_path + 'p*')]

ntuh_partition = {}
ntuh_partition['type'] = 'ntuh'
ntuh_partition['train'] = nh_list
ntuh_partition['validation'] = nc_list

# Create file list for tcia data
tcia_partition = {}
tcia_partition['type'] = 'tcia'
tcia_partition['test'] = []
tcia_partition['train'] = eh_list
tcia_partition['validation'] = []

# Create file list for msd data
msd_partition = {}
msd_partition['type'] = 'msd'
msd_partition['test'] = []
msd_partition['train'] = ec_list
msd_partition['validation'] = []

info = [ntuh_partition, tcia_partition, msd_partition]

# Create box data
save_list = save_boxdata(config, info, mode='train')
save_list_2 = save_boxdata(config, info, mode='validation')

# Save file lists
list_av = [save_list[0], save_list_2[0], save_list[1], save_list[2]]
np.save(result_path, list_av)