import os
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
from time import gmtime, strftime

import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, roc_curve

from utils import load_config, flatten_config_for_logging
from data_loader.data_loader import (convert_csv_to_dict,
                                     load_patches)
from models.net_keras import *
from data_description.visualization import show_train_history

plt.switch_backend('agg')

# Parse Args
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default='./configs/base.yml',
                    type=str, help="train configuration")
parser.add_argument("-r", "--run_name", default=None, type=str,
                    help="run name for this experiment. (Default: time)")
args = parser.parse_args()

# Load config
config = load_config(args.config)
if args.run_name is None:
    config['run_name'] = strftime("%Y%m%d_%H%M%S", gmtime())
else:
    config['run_name'] = args.run_name
pprint(config)

log_path = os.path.join(config['log']['checkpoint_dir'], config['run_name'])
if not os.path.isdir(log_path):
    os.mkdir(log_path)
model_path = os.path.join(config['log']['model_dir'], config['run_name'])
if not os.path.isdir(model_path):
    os.mkdir(log_path)

# Env settings
os.environ["CUDA_VISIBLE_DEVICES"] = config['system']['CUDA_VISIBLE_DEVICES']
gpu_options = tf.GPUOptions(allow_growth=True)
sess_config = tf.ConfigProto(gpu_options=gpu_options)
set_session(tf.Session(config=sess_config))

# split cases into train, val, test
case_partition = convert_csv_to_dict()

# Get patches
patch_size = config['dataset']['input_dim'][0]
train_X, train_y = load_patches(
    config['dataset']['dir'], case_partition['train'], patch_size=patch_size)
print("Finish loading {} patches from {} studies".format(
    train_X.shape[0], len(case_partition['train'])))
valid_X, valid_y = load_patches(
    config['dataset']['dir'],
    case_partition['validation'],
    patch_size=patch_size,
    train_mode=False)
print("Finish loading {} patches from {} studies".format(
    valid_X.shape[0], len(case_partition['validation'])))
test_X, test_y = load_patches(
    config['dataset']['dir'], case_partition['test'], patch_size=patch_size,
    train_mode=False)

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

# Train and evaluate
history = model.fit_generator(
    datagen.flow(train_X, train_y, batch_size=config['train']['batch_size']),
    epochs=config['train']['epochs'],
    callbacks=cbs,
    steps_per_epoch=len(train_X)/config['train']['batch_size'],
    class_weight='auto',
    validation_data=(valid_X, valid_y))

loss, accuracy = model.evaluate(test_X, test_y)
print('loss = ', loss, 'accuracy = ', accuracy)

# Save the result
fig_acc = show_train_history(history, 'acc', 'val_acc')
plt.savefig(os.path.join(log_path, 'acc_plot.png'))

fig_los = show_train_history(history, 'loss', 'val_loss')
plt.savefig(os.path.join(log_path, 'loss_plot.png'))

model.save_weights(os.path.join(model_path, 'weights.h5'))
