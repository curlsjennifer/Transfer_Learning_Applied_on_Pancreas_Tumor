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



config = load_config('./configs/simpleCNN_box_data_mix.yml')
data_path_pancreasct = '/data2/pancreas/box_data/tinghui/Pancreas-CT/'
data_path_msd = '/data2/pancreas/box_data/tinghui/MSD/'

with open('../result/box/jsons/box_0.json' , 'r') as reader:
    case = json.loads(reader.read())
    ntuh_partition, tcia_partition, msd_partition = case['partition']

np.save('/data2/pancreas/box_data/wanyun/patient_list', [ntuh_partition['test'],tcia_partition['test'], msd_partition['test']])