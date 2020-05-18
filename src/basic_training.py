"""
filename : basic_training.py
author : WanYun, Yang (from TingHui, Wu)
date : 2020/04/11
description :
    create a model from source(ntuh) data
"""

import os
import json
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

# Parse Args
plt.switch_backend('agg')
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--run_name", default=None, type=str,
                    help="run name for this experiment. (Default: time)")
parser.add_argument("-d", "--dev", default=None, type=str,
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
[source_train, source_valid, source_test] = np.load(
    name.source_path, allow_pickle=True)
[target_train, target_valid, target_test] = np.load(
    name.target_path, allow_pickle=True)

train_X = source_train.X
train_y = source_train.y
valid_X = source_valid.X
valid_y = source_valid.y

# Data Generators - Keras
datagen = ImageDataGenerator(
    horizontal_flip=True,
    fill_mode='constant',
    cval=0.0,
    vertical_flip=True)
datagen.fit(train_X)

# Model Init
model = eval(config['model']['name'])(config['dataset']['input_dim'])
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
class_weights = class_weight.compute_class_weight(
    'balanced', np.unique(train_y), train_y)

# Model Training
history = model.fit_generator(
    datagen.flow(train_X, train_y, batch_size=config['train']['batch_size']),
    epochs=config['train']['epochs'],
    callbacks=cbs,
    steps_per_epoch=len(train_X) / config['train']['batch_size'],
    class_weight=class_weights,
    validation_data=(valid_X, valid_y))

# Save the results
model.save_weights(name.model_path)

fig_acc = show_train_history(history, 'acc', 'val_acc')
plt.savefig(name.acc_path)

fig_loss = show_train_history(history, 'loss', 'val_loss')
plt.savefig(name.loss_path)
