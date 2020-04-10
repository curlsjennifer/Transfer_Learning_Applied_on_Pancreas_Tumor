import os
import pandas as pd
import numpy as np
import glob
from data_loader.create_json import create_json_cross
from data_loader.data_loader import save_boxdata
from utils import load_config

    
config = load_config('./configs/basic.yml')
exp_name = 'box'
fold = 1
rate = 1 
list_path = '/data2/pancreas/Nifti_data/data_list.csv'
data_path_tcia = '/data2/open_dataset/pancreas/Pancreas-CT/'
data_path_msd = '/data2/open_dataset/MSD/Task07_Pancreas/imagesTr/'

df = pd.read_csv(list_path, converters={'add_date': str})
nh_list = list(df[(df['type'] == 'healthy') & (df['diff_patient_list'])]['case_id'])
nc_list = list(df[(df['type'] == 'tumor') & (df['diff_patient_list'])]['case_id'])

eh_list = [files.split('/')[-1] for files in glob.glob(data_path_tcia + 'P*')]
ec_list = [files.split('/')[-1] for files in glob.glob(data_path_msd + 'p*')]

ntuh_partition = {}
ntuh_partition['type'] = 'ntuh'
ntuh_partition['test'] = []
ntuh_partition['train'] = nh_list
ntuh_partition['validation'] = nc_list


tcia_partition = {}
msd_partition = {}
tcia_partition['type'] = 'tcia'
msd_partition['type'] = 'msd'
tcia_partition['test'] = []
msd_partition['test'] = []

tcia_partition['train'] = eh_list
msd_partition['train'] = ec_list
tcia_partition['validation'] = []
msd_partition['validation'] = []

info = [ntuh_partition, tcia_partition, msd_partition]

save_list = save_boxdata(config, info, mode='train')
save_list_2 = save_boxdata(config, info, mode='validation')

list_av = [save_list[0], save_list_2[0], save_list[1], save_list[2]]
np.save('/data2/pancreas/box_data/wanyun/patient_list.npy', list_av)

print(np.shape(list_av[0]), np.shape(list_av[1]), np.shape(list_av[2]), np.shape(list_av[3]))