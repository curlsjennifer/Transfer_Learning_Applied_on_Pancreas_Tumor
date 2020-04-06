import os
from data_loader.create_json import create_json_cross
from train_mix_basic import train_model
from test_results import test_model
from utils import load_config

config_name = './configs/simpleCNN_box_data_mix.yml'
exp_name = 'cm_100_100'
fold = 5
exp_type = 'mix'

config = load_config(config_name)
create_json_cross(config, exp_name, fold=fold, rate=1, copy='cross_trans_0_25')
exp_path = os.path.join(config['log']['result_dir'], exp_name, 'jsons')

for json_name in os.listdir(exp_path):
    print(json_name.split('.'))
    if exp_type == 'transfer' and json_name==os.listdir(exp_path)[3]:
        print(json_name)
        train_model(config, exp_path + '/' + json_name, json_name.split('.')[0], mode='trans_1')
        train_model(config, exp_path + '/' + json_name, json_name.split('.')[0], mode='trans_2')
    else:
        train_model(config, exp_path + '/' + json_name, json_name.split('.')[0], mode='mix')

test_model(config, exp_name, other_name=True, fold=fold)



