"""
filename : cross_validtion.py
author : WanYun, Yang
date : 2020/04/10
description :
    use cross validation tool to analysis models.
"""

import os

# Fixed variables
exp_type = 'transfer'
target_rate = 1
copy = 'ct_0_100_10'
config_name = './configs/basic.yml'
dev = "'1'"
fold = 10

# Modified variables
fix_layer = 3 # transfer
source_data = True # mix


# Name the experiment
def exp_name(type):
    return {
        'transfer':'_'.join(['ct', str(fix_layer),
                             str(int(target_rate*100)), str(fold)]),
        'mix': '_'.join(['cm', str(int(source_data)*100),
                         str(int(target_rate*100)), str(fold)]),
        'increment':'_'.join(['ci', str(fix_layer),
                             str(int(target_rate*100)), str(fold)]) 
    }[x]
    
if exp_type == 'transfer':
    exp_name = '_'.join(['ct', str(fix_layer), 
                         str(int(target_rate*100)), str(fold)])
else:
    exp_name = '_'.join(['cm', str(int(source_data)*100), 
                         str(int(target_rate*100)), str(fold)])
    
# Source environment
# ENV =  "source /home/u/curlsjennifer/source-python.bashrc"
# os.system("source /home/u/curlsjennifer/source-python.bashrc")
#CMD = "bash -c '" + ENV + "; " + CREATE_JSON + "'"

# Configs and related parameters
os.system(
    "python create_json.py " + \
    "-c " + config_name + " " + \
    "-e '" + exp_name + "' " + \
    "-f " + str(fold) + " " + \
    "-r " + str(target_rate) + " " + \
    "-copy '" + copy + "' "
)

# Transfer learning or mix data for each small model
exp_path = os.path.join("../result/", exp_name, 'jsons')
cross_exp_name = [name.split('.')[0] for name in os.listdir(exp_path)]

check = 0
for json_name in cross_exp_name:
    print("Working on experiment : ", json_name)
    if exp_type == 'transfer' and check == 0:
        check = 1
        # create a initial source model for transfer learning
        # os.system(
        #     "python transfer_1.py "
        #     "-c " + config_name + " "
        #     "-r '" + json_name + "' "
        #     "-j '" + exp_path + "/" + json_name + "' "
        #     "-d " + dev
        # )
        
        # use target data to fine-tune source model
        os.system(
            "python increment.py "
            "-c " + config_name + " "
            "-r '" + json_name + "_trans' "
            "-j '" + exp_path + "/" + json_name + "' "
            "-l " + str(fix_layer) + " "
            "-d " + dev + " "
            "-m '" + os.path.join('../models', json_name, 'weights.h5') + "'"
        )
    else:
        check = 1
        # mix-data model
        # os.system(
        #     "python mix.py "
        #     "-c " + config_name + " "
        #     "-r '" + json_name + "' "
        #     "-s '" + str(source_data) + "' "
        #     "-j '" + exp_path + "/" + json_name + "' "
        #     "-d " + dev
        # )

# Calculate auc for all experiments
# os.system(
#     "python test.py "
#     "-e '" + exp_name + "' "
#     "-t 'none'")
# if exp_type == 'transfer':
#     os.system(
#         "python test.py "
#         "-e '" + exp_name + "' "
#         "-t 'trans2'")
