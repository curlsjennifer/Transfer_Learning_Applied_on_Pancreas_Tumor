"""
filename : transfer_2.py
author : WanYun, Yang (from TingHui, Wu)
date : 2020/04/11
description :
    apply fine-tuning method on model (source) by model (target)
"""

import os
import json
import math
import copy
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix

from utils import load_config, find_threshold, predict_binary
from data_loader.data_loader import get_patches
from data_description.visualization import show_train_history
from net_keras import *

def matrix_R(p1, p2, diff, lambda_1=1, lambda_2=1):
    if diff == 0:
        return -lambda_1 * p1 * math.log(p1)
    else:
        return -lambda_2 * (p1 - p2) * (math.log(p1/p2))
    
def sort_ext(label, ext_X, ext_y, ext_idx, model, alpha=0.2):
    start = sum([id[2] for id in ext_idx[:label]])
    end = start + ext_idx[label][2]
    ext_X = ext_X[start:end]
    ext_y = ext_y[start:end]
    pred = model.predict_proba(ext_X)
    Len = len(pred)
    if np.mean(pred) > 0.5:
        S = sorted(pred)[:int(Len*alpha)]
    else:
        S = sorted(pred)[int(Len*(1-alpha)):]

    S = [[i, S[i][0]] for i in range(len(S))]
    return sum([sum([matrix_R(p1, p2, ind1 - ind2) 
                     for ind1, p1 in S]) for ind2, p2 in S])
    
# Parse Args
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default='./configs/base.yml',
                    type=str, help="train configuration")
parser.add_argument("-r", "--run_name", default=None, type=str,
                    help="run name for this experiment. (Default: time)")
parser.add_argument("-j", "--json", default=None, type=str,
                    help="case json")
parser.add_argument("-l", "--fix_layer", default=None, type=int,
                    help="parameter for cross validation")
parser.add_argument("-m", "--origin_model", default=None, type=str,
                    help="origin_model")  
parser.add_argument("-d", "--dev", default=None, type=str,
                    help="device")                 
args = parser.parse_args()

# Load config
config = load_config(args.config)
config['system']['CUDA_VISIBLE_DEVICES'] = args.dev
exp_name = args.run_name

# Set path
model_path = os.path.join(config['log']['model_dir'], exp_name)
if not os.path.isdir(model_path):
    os.makedirs(model_path)
result_basepath = os.path.join(
    config['log']['result_dir'], exp_name)
if not os.path.isdir(result_basepath):
    os.makedirs(result_basepath)

# Env settings
os.environ["CUDA_VISIBLE_DEVICES"] = config['system']['CUDA_VISIBLE_DEVICES']
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
set_session(tf.compat.v1.Session(config=sess_config))

# Split cases into train, val, test
with open(args.json + '.json' , 'r') as reader:
    case = json.loads(reader.read())
    ntuh_partition, tcia_partition, msd_partition = case['partition']
    ntuh_partition['train'] = []
    ntuh_partition['validation'] = []

# Get train patches
ext_X, ext_y, ext_idx = get_patches(
    config, [ntuh_partition, tcia_partition, msd_partition], mode='train')

ext_X = np.array(ext_X)
ext_X = ext_X.reshape(
    ext_X.shape[0], ext_X.shape[1], ext_X.shape[2], 1)
ext_y = np.array(ext_y)
steps_per_epoch = int(len(ext_y) / config['train']['batch_size'])

# Get valid patches
valid_X, valid_y, valid_idx = get_patches(
    config, [ntuh_partition, tcia_partition, msd_partition], mode='validation')

valid_X = np.array(valid_X)
valid_X = valid_X.reshape(
    valid_X.shape[0], valid_X.shape[1], valid_X.shape[2], 1)
valid_y = np.array(valid_y)

print("Finish loading {} patches from {} studies".format(
    valid_X.shape[0], len(valid_idx)))
print("With {} lesion patches and {} normal pancreas patches".format(
    np.sum(valid_y), valid_X.shape[0] - np.sum(valid_y)))

# Model Init
model = eval(config['model']['name'])(config['dataset']['input_dim'])
model.load_weights(args.origin_model, by_name=True)

model.compile(
    loss=keras.losses.binary_crossentropy,
    optimizer=keras.optimizers.Adam(lr=config['optimizer']['lr'],
                                    amsgrad=True),
    metrics=['accuracy'])
cbs = [
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=10, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=40)
]

# Model Training
class_weights = class_weight.compute_class_weight(
    'balanced', np.unique(ext_y), ext_y)
print("Setting class weights {}".format(class_weights))

item_num = 20
tar_X = np.zeros((0, 50, 50, 1))
tar_y = []
tar_idx = []

for e in range(200):
    print('Epoch', e)
    o_ext_X = copy.copy(ext_X)
    o_ext_y = copy.copy(ext_y)
    o_ext_idx = copy.copy(ext_idx)
    
    ext_X = np.zeros((0, 50, 50, 1))
    ext_y = []
    ext_idx = []
    
    do_X = np.zeros((0, 50, 50, 1))
    do_y = []
    do_idx = []
    
    print(np.shape(o_ext_X), np.shape(o_ext_y), np.shape(o_ext_idx))
    res = [sort_ext(label, o_ext_X, o_ext_y, o_ext_idx, model) 
           for label in range(len(o_ext_idx))]
    
    index = sorted(range(len(res)), key = lambda k : res[k])[:item_num]

    end = 0
    for ind in range(len(o_ext_idx)):
        start = end
        end += o_ext_idx[ind][2]
        if ind in index:
            do_X = np.concatenate((do_X, o_ext_X[start:end, :, :, :]), axis=0)
            do_y.extend(o_ext_y[start:end])
            do_idx.append(o_ext_idx[ind])
        else:
            ext_X = np.concatenate((ext_X, o_ext_X[start:end, :, :, :]), axis=0)
            ext_y.extend(o_ext_y[start:end])
            ext_idx.append(o_ext_idx[ind])
    
    print("NOTE", np.shape(do_X)[0], np.shape(ext_X)[0], np.shape(tar_X)[0], np.shape(o_ext_X)[0])
    model.fit(do_X, np.array(do_y), 
                epochs=10, 
                callbacks=cbs, 
                validation_data=(valid_X, valid_y),
                steps_per_epoch=steps_per_epoch, 
                validation_steps=int(len(valid_X) / config['train']['batch_size']),
                class_weight=class_weights)
    
    tar_X = np.concatenate((tar_X, do_X), axis=0)
    tar_y = np.concatenate((tar_y, do_y), axis=0)
    tar_idx.extend(do_idx)    

model.save_weights(os.path.join(model_path, 'weights.h5'))
