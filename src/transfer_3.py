"""
filename : transfer_2.py
author : WanYun, Yang (from TingHui, Wu)
date : 2020/04/11
description :
    apply fine-tuning method on model (source) by model (target)
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
from data_loader.data_loader import get_patches
from data_description.visualization import show_train_history
from net_keras import *

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
model_path = os.path.join(config['log']['model_dir'], exp_name + '2')
if not os.path.isdir(model_path):
    os.makedirs(model_path)
result_basepath = os.path.join(
    config['log']['result_dir'], exp_name + '2')
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
train_X, train_y, train_idx = get_patches(
    config, [ntuh_partition, tcia_partition, msd_partition], mode='train')

train_X = np.array(train_X)
train_X = train_X.reshape(
    train_X.shape[0], train_X.shape[1], train_X.shape[2], 1)
train_y = np.array(train_y)

print("Finish loading {} patches from {} studies".format(
    train_X.shape[0], len(train_idx)))
print("With {} lesion patches and {} normal pancreas patches".format(
    np.sum(train_y), train_X.shape[0] - np.sum(train_y)))

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

# Data Generators - Keras
datagen = ImageDataGenerator(
    horizontal_flip=True,
    fill_mode='constant',
    cval=0.0,
    vertical_flip=True)
datagen.fit(train_X)

# Model Init
model = eval(config['model']['name'])(config['dataset']['input_dim'])
model.load_weights(args.origin_model, by_name=True)


# Set Trainable Layers
fix_layer = args.fix_layer
i = 0
for layer in model.layers[:fix_layer]:
    layer.trainable = False
    print("False : ", i, layer.name)
    i += 1
for layer in model.layers[fix_layer:]:
    layer.trainable = True
    print("True : ", i, layer.name)
    i += 1

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

# Check error data
valid_probs = model.predict_proba(valid_X)
patch_threshold = find_threshold(valid_probs, valid_y)
train_probs = model.predict_proba(train_X)
probs = predict_binary(train_probs, patch_threshold)
error_index = [i for i in range(len(probs)) if abs(probs[i, 0] - train_y[i]) > 0]
error_train_X = np.repeat(train_X[error_index, :, :, :], 1, axis=0)
error_train_y = np.repeat(train_y[error_index], 1, axis=0)

# Model Training
class_weights = class_weight.compute_class_weight(
    'balanced', np.unique(train_y), train_y)
print("Setting class weights {}".format(class_weights))
print(class_weights)

history = model.fit_generator(
    datagen.flow(error_train_X, error_train_y, batch_size=config['train']['batch_size']),
    epochs=config['train']['epochs'],
    callbacks=cbs,
    steps_per_epoch=len(train_X) / config['train']['batch_size'],
    class_weight=class_weights,
    validation_data=(valid_X, valid_y))

history = model.fit_generator(
    datagen.flow(train_X, train_y, batch_size=config['train']['batch_size']),
    epochs=config['train']['epochs'],
    callbacks=cbs,
    steps_per_epoch=len(train_X) / config['train']['batch_size'],
    class_weight=class_weights,
    validation_data=(valid_X, valid_y))


model.save_weights(os.path.join(model_path, 'weights.h5'))

# Save the result
fig_acc = show_train_history(history, 'acc', 'val_acc')
plt.savefig(os.path.join(result_basepath, 'acc_plot.png'))

fig_los = show_train_history(history, 'loss', 'val_loss')
plt.savefig(os.path.join(result_basepath, 'loss_plot.png'))