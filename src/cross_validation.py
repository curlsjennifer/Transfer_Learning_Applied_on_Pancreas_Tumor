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
fold = 10

# Modified variables
exp_type = 'mix'
target_rate = 0.75
copy = 'ct_0_75_10'
fix_layer = 3
source_data = True
dev = "'1'"

# Name the experiment
if exp_type == 'transfer':
    exp_name = '_'.join(['ct', str(fix_layer), 
                         str(int(target_rate*100)), str(fold)])
else:
    exp_name = '_'.join(['cm', str(int(source_data)*100), 
                         str(int(target_rate*100)), str(fold)])

# Configs and related parameters
os.system(
    "python create_json.py "
    "-c " + config_name + " "
    "-e '" + exp_name + "' "
    "-f " + str(fold) + " "
    "-r " + str(target_rate) + " "
    "-copy '" + copy + "' "
)

# Transfer learning or mix data for each small model
exp_path = os.path.join("../result/", exp_name, 'jsons')
cross_exp_name = [name.split('.')[0] for name in os.listdir(exp_path)]
for json_name in cross_exp_name:
    print("Working on experiment : ", json_name)
    if exp_type == 'transfer':
        # create a initial source model for transfer learning
        os.system(
            "python transfer_1.py "
            "-c " + config_name + " "
            "-r '" + json_name + "' "
            "-j '" + exp_path + "/" + json_name + "' "
            "-d " + dev
        )
        # use target data to fine-tune source model
        os.system(
            "python transfer_2.py "
            "-c " + config_name + " "
            "-r '" + json_name + "_trans' "
            "-j '" + exp_path + "/" + json_name + "' "
            "-l " + str(fix_layer) + " "
            "-d " + dev + " "
            "-m '" + os.path.join('../models', json_name, 'weights.h5') + "'"
        )
    else:
        # mix-data model
        os.system(
            "python mix.py "
            "-c " + config_name + " "
            "-r '" + json_name + "' "
            "-s '" + str(source_data) + "' "
            "-j '" + exp_path + "/" + json_name + "' "
            "-d " + dev
        )

# Calculate auc for all experiments
os.system(
    "python test.py "
    "-e '" + exp_name + "' "
    "-t 'none'")
if exp_type == 'transfer':
    os.system(
        "python test.py "
        "-e '" + exp_name + "' "
        "-t 'trans2'")
