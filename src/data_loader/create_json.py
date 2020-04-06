import glob
import os
import copy
import json
from pandas import DataFrame

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import nibabel as nib
import random
from random import shuffle
from shutil import copyfile, rmtree
import SimpleITK as sitk


from data_loader.patch_sampler import patch_generator
from data_loader.patch_sampler import masked_2D_sampler
from data_loader.preprocessing import (
    minmax_normalization, windowing, smoothing)
from data_loader.create_boxdata import finecut_to_thickcut
from utils import load_config, flatten_config_for_logging, find_threshold, predict_binary


def create_json_cross(config, exp_name, fold=5, rate=1, copy=None):


    # data_path_pancreasct = '/data2/pancreas/box_data/tinghui/Pancreas-CT/'
    # data_path_msd = '/data2/pancreas/box_data/tinghui/MSD/'
    
    exp_path = os.path.join(config['log']['result_dir'], exp_name, 'jsons')
    tvt_path = os.path.join(config['log']['result_dir'], exp_name, 'tvt')
    if os.path.isdir(exp_path):
        rmtree(exp_path)
    os.makedirs(exp_path)
    if os.path.isdir(tvt_path):
        rmtree(tvt_path)
    os.makedirs(tvt_path)

    if copy:
        if "json" in copy:
            print('Testing jsons files......')
            with open(os.path.join(config['log']['checkpoint_dir'], copy) , 'r') as reader:
                case = json.loads(reader.read())

            info = {}
            info['partition'] = case['partition']
            with open(os.path.join(config['log']['result_dir'], exp_name, 'jsons', exp_name + '_0' + '.json'), 'w') as f:
                json.dump(info, f)
            
            # write info
            ntuh_partition, tcia_partition, msd_partition = info['partition']
            data={
            'title':['train', 'validation', 'test'],
            'ntuh_health':[len(ntuh_partition['train'])/2,len(ntuh_partition['validation'])/2,len(ntuh_partition['test'])/2],
            'ntuh_cancer':[len(ntuh_partition['train'])/2,len(ntuh_partition['validation'])/2,len(ntuh_partition['test'])/2],
            'ext_health':[len(tcia_partition['train']),len(tcia_partition['validation']),len(tcia_partition['test'])],
            'ext_cancer':[len(msd_partition['train']),len(msd_partition['validation']),len(msd_partition['test'])]
            }
            df=DataFrame(data)
            df.to_csv(os.path.join(tvt_path, exp_name + '_0_info.csv'))

        else:
            print('Copying jsons files......')
            origin_path = os.path.join(config['log']['result_dir'], copy, 'jsons')
            for json_file in os.listdir(origin_path):
                copyfile(origin_path + '/' + json_file, exp_path + '/' + json_file.replace(copy, exp_name))
            origin_path = os.path.join(config['log']['result_dir'], copy, 'tvt')
            for csv_file in os.listdir(origin_path):
                copyfile(origin_path + '/' + csv_file, tvt_path + '/' + csv_file.replace(copy, exp_name))  

    else:
        print('Creating jsons files......')
        list_path = config['dataset']['csv']
        # df = pd.read_csv(list_path, converters={'add_date': str})

        # nh_list = list(df[(df['type'] == 'healthy') & (df['diff_patient_list'])]['case_id'])
        # nc_list = list(df[(df['type'] == 'tumor') & (df['diff_patient_list'])]['case_id'])
        # # nh_list = nh_list[:400]
        # # nc_list = nc_list[:400]
        # eh_list = os.listdir(data_path_pancreasct)
        # print(data_path_pancreasct)
        # ec_list = os.listdir(data_path_msd)
        # eh_list = [file for file in eh_list if file[0] == 'P']
        # ec_list = [file for file in ec_list if file[0] == 'p']

        [nh_list, nc_list, eh_list, ec_list] = np.load('/data2/pancreas/box_data/wanyun/patient_list.npy', allow_pickle=True)

        test_num_h = int(np.around(len(nh_list)/fold))
        test_num_c = int(np.around(len(nc_list)/fold))
        val_num_h = int(np.around((len(nh_list)-test_num_h)*0.1))
        val_num_c = int(np.around((len(nc_list)-test_num_c)*0.1))

        test_num_h_e = int(np.around(len(eh_list)/fold))
        test_num_c_e = int(np.around(len(ec_list)/fold))
        val_num_h_e = int(np.around((len(eh_list)-test_num_h_e)*0.1*rate))
        val_num_c_e = int(np.around((len(ec_list)-test_num_c_e)*0.1*rate))
        for index in range(fold):
            ntuh_partition = {}
            ntuh_partition['type'] = 'ntuh'
            ntuh_partition['test'] = nh_list[test_num_h*(index):test_num_h*(index+1)] \
                + nc_list[test_num_c*(index):test_num_c*(index+1)]

            tv_h = [x for x in nh_list if x not in ntuh_partition['test']]
            random.Random(config['dataset']['seed']).shuffle(tv_h)
            tv_c = [x for x in nc_list if x not in ntuh_partition['test']]
            random.Random(config['dataset']['seed']).shuffle(tv_c)

            ntuh_partition['train'] = tv_h[val_num_h:] + tv_c[val_num_c:]
            ntuh_partition['validation'] = tv_h[:val_num_h] + tv_c[:val_num_c]


            tcia_partition = {}
            msd_partition = {}
            tcia_partition['type'] = 'tcia'
            msd_partition['type'] = 'msd'
            tcia_partition['test'] = eh_list[test_num_h_e*(index):test_num_h_e*(index+1)] 
            msd_partition['test'] = ec_list[test_num_c_e*(index):test_num_c_e*(index+1)]

            tv_h = [x for x in eh_list if x not in tcia_partition['test']] 
            random.Random(config['dataset']['seed']).shuffle(tv_h)
            tv_h = tv_h[:int(np.around(len(tv_h)*rate))]
            tv_c = [x for x in ec_list if x not in msd_partition['test']]
            random.Random(config['dataset']['seed']).shuffle(tv_c)
            tv_c = tv_c[:int(np.around(len(tv_c)*rate))] 

            tcia_partition['train'] = tv_h[val_num_h_e:]
            msd_partition['train'] = tv_c[val_num_c_e:]
            tcia_partition['validation'] = tv_h[:val_num_h_e]
            msd_partition['validation'] = tv_c[:val_num_c_e]
            
            info = {}
            info['partition'] = [ntuh_partition, tcia_partition, msd_partition]
            with open(os.path.join(config['log']['result_dir'], exp_name, 'jsons', exp_name + '_' + str(index) + '.json'), 'w') as f:
                json.dump(info, f)
            
            # write info

            data={
            'title':['train', 'validation', 'test'],
            'ntuh_health':[len(ntuh_partition['train'])/2,len(ntuh_partition['validation'])/2,len(ntuh_partition['test'])/2],
            'ntuh_cancer':[len(ntuh_partition['train'])/2,len(ntuh_partition['validation'])/2,len(ntuh_partition['test'])/2],
            'ext_health':[len(tcia_partition['train']),len(tcia_partition['validation']),len(tcia_partition['test'])],
            'ext_cancer':[len(msd_partition['train']),len(msd_partition['validation']),len(msd_partition['test'])]
            }
            df=DataFrame(data)
            df.to_csv(os.path.join(tvt_path, exp_name + '_' + str(index) + '_info.csv'))

