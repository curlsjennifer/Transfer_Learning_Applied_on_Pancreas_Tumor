import os
from data_loader.create_json import create_json_cross
from utils import load_config

# fixed
config_name = './configs/basic.yml'
fold = 10

# modify
exp_name = 'ct_0_25_10'
exp_type = 'transfer'
dev = "'0'"

config = load_config(config_name)
create_json_cross(config, exp_name, fold=fold, rate=1)
exp_path = os.path.join(config['log']['result_dir'], exp_name, 'jsons')

state_t1 = "python transfer_1.py -c ./configs/basic.yml -r '"
state_t2 = "python transfer_2.py -c ./configs/basic.yml -r '"
state_mix = "python mix.py -c ./configs/basic.yml -r '"

for json_name in os.listdir(exp_path):
    name = json_name.split('.')[0]
    state_dev = name + "' -d " + dev
    state_j1 = "' -j '" + exp_path + '/'
    state_j2 = "_trans' -j '" + exp_path + '/'
    model_path = config['log']['model_dir']
    state_model = "' -m '" + os.path.join(model_path, name, 'weights.h5')

    print(name)
    if exp_type == 'transfer':
        os.system(state_t1 + name + state_j1 + state_dev)
        os.system(state_t2 + name + state_j2 + state_dev + state_model)
    else:
        os.system(state_mix + name + state_j1 + state_dev)

os.system("python test.py -r '" + exp_name + "' -t 'none'")
if exp_type == 'transfer':
    os.system("python test.py -r '" + exp_name + "' -t 'trans'")
