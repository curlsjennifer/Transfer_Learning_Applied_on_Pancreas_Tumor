"""
filename : create_json
author : WanYun, Yang
date : 2020/04/11
description :
    use create .json files to create several lists for
    train/validation/test set.
"""

import os
import json
import random
import numpy as np
import pandas as pd
from random import shuffle
from pandas import DataFrame
from shutil import copyfile, rmtree


def create_json_cross(config, exp_name, fold=5, rate=1, copy=None):
    """
    Description:
        use create .json files to create several lists for 
        train/validation/test sets.

    Args:
        config: config, including file paths.
        exp_name: experiment name.
        fold: number of folds for cross validation.
        rate: rate to control the amount of target data. If rate = 1, use all
        target data; if rate = 0, use 0 target data.
        copy: copy .json files from other experiments
    """

    # Create folders for .json files
    exp_path = os.path.join('../result', exp_name, 'jsons')
    list_location = '/data2/pancreas/box_data/wanyun/patient_list.npy'
    if os.path.isdir(exp_path):
        rmtree(exp_path)
    os.makedirs(exp_path)

    if copy:
        # Copy .json files
        print('Copying jsons files......')
        origin_path = os.path.join('../result', copy, 'jsons')
        for json_file in os.listdir(origin_path):
            copyfile(origin_path + '/' + json_file,
                     exp_path + '/' + json_file.replace(copy, exp_name))

    else:
        # Create .json files
        print('Creating jsons files......')
        [nh_list, nc_list, eh_list, ec_list] = np.load(list_location,
                                                       allow_pickle=True)

        # Calculate parameters for spliting train/validation/test sets
        test_num_h = int(np.around(len(nh_list) / fold))
        test_num_c = int(np.around(len(nc_list) / fold))
        val_num_h = int(np.around((len(nh_list) - test_num_h) * 0.1))
        val_num_c = int(np.around((len(nc_list) - test_num_c) * 0.1))

        test_num_h_e = int(np.around(len(eh_list) / fold))
        test_num_c_e = int(np.around(len(ec_list) / fold))
        val_num_h_e = int(np.around((len(eh_list) - test_num_h_e) * 0.1 * rate))
        val_num_c_e = int(np.around((len(ec_list) - test_num_c_e) * 0.1 * rate))

        # create .json files for each experiments
        for index in range(fold):
            ntuh_partition = {}
            ntuh_partition['type'] = 'ntuh'
            ntuh_partition['test'] = nh_list[
                test_num_h * (index):test_num_h * (index + 1)]
            + nc_list[
                test_num_c * (index):test_num_c * (index + 1)]

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
            tcia_partition['test'] = eh_list[
                test_num_h_e * (index):test_num_h_e * (index + 1)]
            msd_partition['test'] = ec_list[
                test_num_c_e * (index):test_num_c_e * (index + 1)]

            tv_h = [x for x in eh_list if x not in tcia_partition['test']]
            random.Random(config['dataset']['seed']).shuffle(tv_h)
            tv_h = tv_h[:int(np.around(len(tv_h) * rate))]
            tv_c = [x for x in ec_list if x not in msd_partition['test']]
            random.Random(config['dataset']['seed']).shuffle(tv_c)
            tv_c = tv_c[:int(np.around(len(tv_c) * rate))]

            tcia_partition['train'] = tv_h[val_num_h_e:]
            msd_partition['train'] = tv_c[val_num_c_e:]
            tcia_partition['validation'] = tv_h[:val_num_h_e]
            msd_partition['validation'] = tv_c[:val_num_c_e]

            # Save files
            info = {}
            info['partition'] = [ntuh_partition, tcia_partition, msd_partition]

            path_split = ['../result/', exp_name,
                          'jsons', exp_name + '_' + str(index) + '.json']
            with open(os.path.join(*path_split), 'w') as f:
                json.dump(info, f)
