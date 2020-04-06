import os
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
from time import gmtime, strftime, localtime
import pandas as pd
from random import shuffle
import logging
import absl.logging as absl_log
import json
# from comet_ml import Experiment

import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, roc_curve, auc

from utils import load_config, flatten_config_for_logging, find_threshold, predict_binary
from data_loader.data_loader import (DataGenerator, generate_case_partition,
                                     get_patches, load_patches, save_boxdata)
from models.net_keras import *
from data_description.visualization import plot_roc, show_train_history

plt.switch_backend('agg')

# Parse Args
parser = argparse.ArgumentParser()
args = parser.parse_args()

# Load config
config = load_config('./configs/simpleCNN_box_data_mix.yml')
config['system']['CUDA_VISIBLE_DEVICES'] = '1'

# Env settings
os.environ["CUDA_VISIBLE_DEVICES"] = config['system']['CUDA_VISIBLE_DEVICES']
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
set_session(tf.compat.v1.Session(config=sess_config))

# split cases into train, val, test
#ntuh_partition, tcia_partition, msd_partition = generate_case_partition(config)
with open('../result/box/jsons/box_0.json' , 'r') as reader:
    case = json.loads(reader.read())
    ntuh_partition, tcia_partition, msd_partition = case['partition']

logging.info("Finish data partition")

# Get train patches
train_X, train_y, train_idx = save_boxdata(
    config, [ntuh_partition, tcia_partition, msd_partition], mode='test')