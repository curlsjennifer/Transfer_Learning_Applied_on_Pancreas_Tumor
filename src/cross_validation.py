"""
filename : cross_validtion.py
author : WanYun, Yang
date : 2020/04/10
description :
    use cross validation tool to analysis models.
"""

import os

# Fixed variables
config_name = './configs/basic.yml'
dev = "'1'"
fold = 10

# Modified variables
fix_layer = 3
box_data_index = 0
exp_name = 'inc_R_t'
num_tar = 200

# Create folders
model_path = os.path.join('../results', exp_name, 'models')
if not os.path.isdir(model_path):
    os.makedirs(model_path)
    os.makedirs(model_path.replace('models', 'acc'))
    os.makedirs(model_path.replace('models', 'loss'))
    os.makedirs(model_path.replace('models', 'time'))
    os.makedirs(model_path.replace('models', 'source_rocs'))
    os.makedirs(model_path.replace('models', 'target_rocs'))
    os.makedirs(model_path.replace('models', 'source_patch'))
    os.makedirs(model_path.replace('models', 'target_patch'))

# Transfer learning or mix data for each small model 
cross_exp_name = ['_'.join([exp_name, str(box_data_index), str(ind), str(num_tar)])
                   for ind in range(fold)]

for data_name in cross_exp_name:
    print("Working on experiment : ", data_name)
    # create a initial source model for transfer learning
    os.system(
        "python increment.py -t 'random' "
        "-r '" + data_name + "' "
        "-d " + dev
    )
