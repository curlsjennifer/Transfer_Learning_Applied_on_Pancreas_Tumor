import os
from ast import literal_eval
from pprint import pprint
from random import shuffle

import numpy as np
from tqdm import tqdm, trange

import tensorflow as tf

import keras
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator

from data_loader.data_loader import (fix_save_case_partition,
                                     load_case_partition, load_patches)
from models.net_keras import simple_cnn_sigmoid_keras
from utils import get_config_sha1

# Load config
with open('./configs/config_tinghui.txt', 'r') as f:
    config = literal_eval(f.read())
    config['config_sha1'] = get_config_sha1(config, 5)
    pprint(config)

os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_VISIBLE_DEVICES']
gpu_options = tf.GPUOptions(allow_growth=True)
sess_config = tf.ConfigProto(gpu_options=gpu_options)

set_session(tf.Session(config=sess_config))

# split cases into train, val, test
case_list = os.listdir(config['case_list_dir'])
case_partition = fix_save_case_partition(
    case_list,
    config['case_split_ratio'],
    path=config['case_partition_path'],
    random_seed=config['random_seed']
) if config['case_partition_path'] == '' else load_case_partition(
    config['case_partition_path'])

# Get patches
patch_size = config['input_dim'][0]
train_X, train_y = load_patches(
    config['case_list_dir'], case_partition['train'], patch_size=patch_size)
print("Finish loading {} patches from {} studies".format(
    train_X.shape[0], len(case_partition['train'])))
valid_X, valid_y = load_patches(
    config['case_list_dir'],
    case_partition['validation'],
    patch_size=patch_size)
print("Finish loading {} patches from {} studies".format(
    valid_X.shape[0], len(case_partition['validation'])))
test_X, test_y = load_patches(
    config['case_list_dir'], case_partition['test'], patch_size=patch_size)

# Data Generators - Keras
datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='constant',
    cval=0.0,
    vertical_flip=True)
datagen.fit(train_X)

# Model Init - Ignored learning rate settings.
model = simple_cnn_sigmoid_keras(config['input_dim'])
model.compile(
    loss=keras.losses.binary_crossentropy,
    optimizer=keras.optimizers.Adam(amsgrad=True),
    metrics=['accuracy'])
cbs = [
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=10, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=40)
]

history = model.fit_generator(
    datagen.flow(train_X, train_y, batch_size=32),
    epochs=config["epochs"],
    callbacks=cbs,
    steps_per_epoch=len(train_X) / 32,
    class_weight='auto',
    validation_data=(valid_X, valid_y))

loss, accuracy = model.evaluate(test_X, test_y)
print('loss = ', loss, 'accuracy = ', accuracy)
