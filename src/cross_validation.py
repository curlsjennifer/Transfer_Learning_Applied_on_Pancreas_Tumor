"""
filename : cross_validtion
author : WanYun, Yang
date : 2020/04/10
description : use cross validation tool to analysis models
"""

import os
from utils import load_config
from data_loader.create_json import create_json_cross

# Fixed variables
config_name = './configs/basic.yml'
fold = 10


# Modified variables
exp_name = 'ct_0_25_10'
exp_type = 'transfer'
dev = "'0'"


# Configs and related parameters
config = load_config(config_name)
create_json_cross(config, exp_name, fold=fold, rate=1)
exp_path = os.path.join(config['log']['result_dir'], exp_name, 'jsons')
model_path = config['log']['model_dir']
cross_exp_name = [name.split('.')[0] for name in os.listdir(exp_path)]


# Transfer learning or mix data for each small model
for json_name in cross_exp_name:
    print("Working on experiment : ", json_name)
    if exp_type == 'transfer':
        # create a initial source model for transfer learning 
        os.system(
            "python transfer_1.py "
            "-c " + config_name + " "
            "-r '" + json_name + "' "
            "-j '" + exp_path + "/" + exp_name + "' "
            "-d " + dev
        )
        # use target data to fine-tune source model
        os.system(
            "python transfer_2.py "
            "-c " + config_name + " "
            "-r '" + json_name + "_trans' "
            "-j '" + exp_path + "/" + exp_name + "' "
            "-d " + dev + " "
            "-m '" + os.path.join(model_path, name, 'weights.h5')
        )
    else:
        # mix-data model
        os.system(
            "python mix.py "
            "-c " + config_name + " "
            "-r '" + json_name + "' "
            "-j '" + exp_path + "/" + exp_name + "' "
            "-d " + dev
        )


# Calculate auc for all experiments
os.system(
    "python test.py "
    "-r '" + exp_name + "' "
    "-t 'none'"
    )
if exp_type == 'transfer':
    os.system("python test.py "
    "-r '" + exp_name + "' 
    "-t 'trans'"
    )
