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
import random
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
from data_loader.data_loader import get_patches, dataset, exp_path
from data_description.visualization import show_train_history
from net_keras import *

def matrix_R(p1, p2, diff, lambda_1=1, lambda_2=1):
    if p1 == 0:
        return 0
    elif diff == 0:
        return -lambda_1 * p1 * math.log(p1)
    else:
        return -lambda_2 * (p1 - p2) * (math.log(p1/p2))
    
def sort_ext(label, ext_X, ext_y, ext_idx, model, alpha=0.25):
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
plt.switch_backend('agg')
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--run_name", default=None, type=str,
                    help="run name for this experiment. (Default: time)")
parser.add_argument("-d", "--dev", default=None, type=str,
                    help="device")
parser.add_argument("-t", "--type", default="selected", type=str,
                    help="device")
parser.add_argument("-n", "--num", default=40, type=int,
                    help="device")
args = parser.parse_args()

# Env settings
os.environ["CUDA_VISIBLE_DEVICES"] = args.dev
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
set_session(tf.compat.v1.Session(config=sess_config))

# Load config and set paths
config = load_config('./configs/basic.yml')
name = exp_path(args.run_name)

# Load training and validation data
[target_train, target_valid, target_test] = np.load(
    name.target_path, allow_pickle=True)

ext_X = target_train.X
ext_y = target_train.y
ext_idx = target_train.idx
valid_X = target_valid.X
valid_y = target_valid.y
valid_idx = target_valid.idx

# Model Init
model = eval(config['model']['name'])(config['dataset']['input_dim'])
model.load_weights(name.weight_path, by_name=True)

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

item_num = int(args.num * 0.9)
tar_X = np.zeros((0, 50, 50, 1))
tar_y = []
tar_idx = []
steps_per_epoch = int(np.shape(ext_X)[0] / config['train']['batch_size'])
history_list = []

while np.shape(ext_X)[0] > 0:
    o_ext_X = copy.copy(ext_X)
    o_ext_y = copy.copy(ext_y)
    o_ext_idx = copy.copy(ext_idx)

    ext_X = np.zeros((0, 50, 50, 1))
    ext_y = []
    ext_idx = []

    do_X = np.zeros((0, 50, 50, 1))
    do_y = []
    do_idx = []

    res = [sort_ext(label, o_ext_X, o_ext_y, o_ext_idx, model) 
            for label in range(len(o_ext_idx))]

    if args.type == "selected":
        index = sorted(range(len(res)), key = lambda k : res[k])[:item_num]
    else:
        index = random.sample(list(range(len(res))), item_num)

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

    tar_X = np.concatenate((tar_X, do_X), axis=0)
    tar_y = np.concatenate((tar_y, do_y), axis=0)
    tar_idx.extend(do_idx)    

    print("Num of do : ", np.shape(do_idx)[0],
            "\nNum of ext : ", np.shape(ext_idx)[0], 
            "\nNum of tar : ", np.shape(tar_idx)[0], 
            "\nNum of prev ext : ", np.shape(o_ext_idx)[0])

    # model = eval(config['model']['name'])(config['dataset']['input_dim'])
    # model.load_weights(name.weight_path, by_name=True)
    # model.compile(
    #     loss=keras.losses.binary_crossentropy,
    #     optimizer=keras.optimizers.Adam(lr=config['optimizer']['lr'],
    #                                     amsgrad=True),
    #     metrics=['accuracy'])
    
    datagen = ImageDataGenerator(
    horizontal_flip=True,
    fill_mode='constant',
    cval=0.0,
    vertical_flip=True)
    datagen.fit(tar_X)
    history = model.fit_generator(
        datagen.flow(tar_X, np.array(tar_y), batch_size=config['train']['batch_size']),
        #epochs=config['train']['epochs'],
        epochs = 66,
        callbacks=cbs,
        steps_per_epoch=steps_per_epoch,
        class_weight=class_weights,
        validation_data=(valid_X, valid_y))
    history_list.append(history.history)

    tar_len = str(int(np.round(np.shape(tar_idx)[0]/0.9)))
    model.save_weights(name.model_path.replace(
        name.dataset_index, tar_len + '_' + name.dataset_index))
    np.save(name.roc_history, history_list)
