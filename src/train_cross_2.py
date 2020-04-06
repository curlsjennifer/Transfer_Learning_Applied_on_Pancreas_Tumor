import os
from data_loader.create_json import create_json_cross
from utils import load_config

config_name = './configs/simpleCNN_box_data_mix.yml'
fold = 10

# modify
exp_name = 'ct_0_25_10'
exp_type = 'transfer'
dev = "'1'"

config = load_config(config_name)
create_json_cross(config, exp_name, fold=fold, rate=1)
# exp_path = os.path.join(config['log']['result_dir'], exp_name, 'jsons')

# for json_name in os.listdir(exp_path):
#     name = json_name.split('.')[0]
#     print(name)
#     if exp_type == 'transfer':
#         os.system("python transfer_1.py -c ./configs/simpleCNN_box_data_mix.yml -r '" + \
#         name +"' -j '"+ exp_path + '/' + name + "' -d " + dev)
#         os.system("python transfer_2.py -c ./configs/simpleCNN_box_data_mix.yml -r '" + \
#         name +"_trans' -j '"+ exp_path + '/' + name + "' -m '" + os.path.join(
#         config['log']['model_dir'], name, 'weights.h5') + "' -d " + dev) 
#     else:
#         os.system("python mix.py -c ./configs/simpleCNN_box_data_mix.yml -r '" + \
#         name +"' -j '"+ exp_path + '/' + name + "' -d " + dev)

# os.system("python test.py -r '" + exp_name + "' -t 'none'")
# if exp_type == 'transfer':
#     os.system("python test.py -r '" + exp_name + "' -t 'trans'")

