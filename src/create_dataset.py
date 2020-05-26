"""
filename : create_dataset.py
author : WanYun, Yang
date : 2020/05/13
description :
    use create .json files to generate train/valid/test
    dataser for cross validation.
Use : python create_dataset.py -f 10 -p '/data2/pancreas/box_data/wanyun/cv_10/' -n 50
"""

import os
import json
import random
import argparse
import numpy as np
import pandas as pd
from random import shuffle
from pandas import DataFrame
from shutil import copyfile, rmtree

from utils import load_config
from data_loader.data_loader import get_patches, dataset

# class dataset:
#     def __init__(self, info, mode):
#         onfig = load_config('./configs/basic.yml')
#         X, y, idx, coord = get_patches(config, info, mode=mode)
#         X = np.array(X)
#         X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
#         y = np.array(y)
#         self.X = X
#         self.y = y
#         self.idx = idx
#         self.coord = coord
#         self.mode = mode
    
# Parse Args
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fold", default=None, type=int,
                    help="fold for cross validation") 
parser.add_argument("-p", "--dataset_path", default=None, type=str,
                    help="path to save files")  
parser.add_argument("-n", "--num_of_ext", default=None, type=int,
                    help="number of ext file per step")             
args = parser.parse_args()
fold = args.fold
num_of_ext = args.num_of_ext
dataset_path = args.dataset_path
config = load_config('./configs/basic.yml')

# Create folders for .json files
list_location = '/data2/pancreas/box_data/wanyun/patient_list.npy'

# Create .json files
print('Creating jsons files......')
[nh_list, nc_list, eh_list, ec_list] = np.load(list_location,
                                                allow_pickle=True)
print(np.shape(nh_list), np.shape(nc_list), np.shape(eh_list), np.shape(ec_list))
# Calculate parameters for spliting train/validation/test sets
test_num_h = int(np.around(len(nh_list) / fold))
test_num_c = int(np.around(len(nc_list) / fold))
val_num_h = int(np.around((len(nh_list) - test_num_h) * 0.1))
val_num_c = int(np.around((len(nc_list) - test_num_c) * 0.1))

test_num_h_e = int(np.around(len(eh_list) / fold))
test_num_c_e = int(np.around(len(ec_list) / fold))
health_rate = len(eh_list) / (len(eh_list) + len(ec_list))
num_of_step = int((len(eh_list) + len(ec_list))
                  * (1 - 1 / fold) / num_of_ext)

# create .json files for each experiments
for index in range(fold):
    ntuh_partition = {}
    ntuh_partition['type'] = 'ntuh'
    ntuh_partition['test'] = nh_list[test_num_h * (index):test_num_h * (index + 1)] \
    + nc_list[test_num_c * (index):test_num_c * (index + 1)]

    tv_h = [x for x in nh_list if x not in ntuh_partition['test']]
    random.Random(config['dataset']['seed']).shuffle(tv_h)
    tv_c = [x for x in nc_list if x not in ntuh_partition['test']]
    random.Random(config['dataset']['seed']).shuffle(tv_c)

    ntuh_partition['train'] = tv_h[val_num_h:] + tv_c[val_num_c:]
    ntuh_partition['validation'] = tv_h[:val_num_h] + tv_c[:val_num_c]

    source_train = dataset([ntuh_partition], 'train')
    source_valid = dataset([ntuh_partition], 'validation')
    source_test = dataset([ntuh_partition], 'test')
    source_dataset = [source_train, source_valid, source_test]
    
    np.save(os.path.join(dataset_path, "source_" + str(index)), source_dataset)
    
    tcia_partition = {}
    msd_partition = {}
    tcia_partition['type'] = 'tcia'
    msd_partition['type'] = 'msd'
    tcia_partition['test'] = eh_list[
        test_num_h_e * (index):test_num_h_e * (index + 1)]
    msd_partition['test'] = ec_list[
        test_num_c_e * (index):test_num_c_e * (index + 1)]

    tv_h = [x for x in eh_list if x not in tcia_partition['test']]
    random.Random(config['dataset']['seed']).shuffle(tv_h)
    t_h = tv_h[:int(np.around(len(tv_h)*0.9))]
    v_h = tv_h[int(np.around(len(tv_h)*0.9)):]
    
    tv_c = [x for x in ec_list if x not in msd_partition['test']]
    random.Random(config['dataset']['seed']).shuffle(tv_c)
    t_c = tv_c[:int(np.around(len(tv_c)*0.9))]
    v_c = tv_c[int(np.around(len(tv_c)*0.9)):]

    for ind in range(num_of_step):
        num_t_h = int(np.round(health_rate * 0.9 * (ind + 1) * num_of_ext))
        num_t_c = int(np.round((1 - health_rate) * 0.9 * (ind + 1) * num_of_ext))
        num_v_h = int(np.round(health_rate * 0.1 * (ind + 1) * num_of_ext))
        num_v_c = num_of_ext * (ind + 1) - num_t_h - num_t_c - num_v_h

        tcia_partition['train'] = t_h[:num_t_h]
        msd_partition['train'] = t_c[:num_t_c]
        tcia_partition['validation'] = v_h[:num_v_h]
        msd_partition['validation'] = v_c[:num_v_c]

        target_train = dataset([tcia_partition, msd_partition], 'train')
        target_valid = dataset([tcia_partition, msd_partition], 'validation')
        target_test = dataset([tcia_partition, msd_partition], 'test')
        target_dataset = [target_train, target_valid, target_test]
    
        np.save(os.path.join(dataset_path,
                             "target_" + str(index) 
                             + "_" + str((ind + 1) * num_of_ext)), 
                target_dataset)
        
        
        
        
        
        
