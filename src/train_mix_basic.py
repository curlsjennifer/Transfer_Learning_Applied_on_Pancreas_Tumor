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
                                     get_patches_list, load_patches)
from models.net_keras import *
from data_description.visualization import plot_roc, show_train_history

plt.switch_backend('agg')

def train_model(config, json_name, exp_name, mode='mix'):

    # Set path
    model_path = os.path.join(config['log']['result_dir'], '_'.join(exp_name.split('_')[:-1]), 'models')
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    result_path = os.path.join(config['log']['result_dir'], '_'.join(exp_name.split('_')[:-1]), 'figures')
    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    # Env settings
    os.environ["CUDA_VISIBLE_DEVICES"] = config['system']['CUDA_VISIBLE_DEVICES']
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    sess_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    set_session(tf.compat.v1.Session(config=sess_config))
    
    # Load case partition
    with open(json_name , 'r') as reader:
        case = json.loads(reader.read())

    # Get train patches
    if mode=='trans_1':
        case['partition'][1]['train'] = []
        case['partition'][2]['train'] = []
        case['partition'][1]['validation'] = []
        case['partition'][2]['validation'] = []

    if mode=='trans_2':
        case['partition'][0]['train'] = []
        case['partition'][0]['validation'] = []

    train_X, train_y, train_idx = get_patches_list(config, case['partition'], mode='train')

    train_X = np.array(train_X)
    train_X = train_X.reshape(
        train_X.shape[0], train_X.shape[1], train_X.shape[2], 1)
    train_y = np.array(train_y)
    print("Finish loading training data.")
    # print("Finish loading {} patches from {} studies".format(
    #     train_X.shape[0], len(train_idx)))
    # logging.info("Finish loading {} patches from {} studies".format(
    #     train_X.shape[0], len(train_idx)))
    # print("With {} lesion patches and {} normal pancreas patches".format(
    #     np.sum(train_y), train_X.shape[0] - np.sum(train_y)))
    # logging.info("With {} lesion patches and {} normal pancreas patches".format(
    #     np.sum(train_y), train_X.shape[0] - np.sum(train_y)))

    # Get valid patches
    valid_X, valid_y, valid_idx = get_patches_list(config, case['partition'], mode='validation')

    valid_X = np.array(valid_X)
    valid_X = valid_X.reshape(
        valid_X.shape[0], valid_X.shape[1], valid_X.shape[2], 1)
    valid_y = np.array(valid_y)
    print("Finish loading validation data.")
    # print("Finish loading {} patches from {} studies".format(
    #     valid_X.shape[0], len(valid_idx)))
    # logging.info("Finish loading {} patches from {} studies".format(
    #     valid_X.shape[0], len(valid_idx)))
    # print("With {} lesion patches and {} normal pancreas patches".format(
    #     np.sum(valid_y), valid_X.shape[0] - np.sum(valid_y)))
    # logging.info("With {} lesion patches and {} normal pancreas patches".format(
    #     np.sum(valid_y), valid_X.shape[0] - np.sum(valid_y)))

    print(np.shape(train_X))
    print(np.shape(valid_X))
    # Data Generators - Keras
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        fill_mode='constant',
        cval=0.0,
        vertical_flip=True)
    datagen.fit(train_X)


# Model Init
    model = eval(config['model']['name'])(config['dataset']['input_dim'])

    if mode == 'trans_2':
        
        model.load_weights(os.path.join(model_path, exp_name + '.h5'))
        print(os.path.join(model_path, exp_name + '.h5'))

        # Set Trainable Layers
        fix_layer = config['model']['fix_layer']
        i = 0
        for layer in model.layers[:fix_layer]:
            layer.trainable=False
            print("False : ", i, layer.name)
            i += 1
        for layer in model.layers[fix_layer:]:
            layer.trainable=True
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
    
    
    # Model Training
    class_weights = class_weight.compute_class_weight(
        'balanced', np.unique(train_y), train_y)
    print("Setting class weights {}".format(class_weights))
    logging.info("Setting class weights {}".format(class_weights))

    logging.info("Start training")
    history = model.fit_generator(
        datagen.flow(train_X, train_y, batch_size=config['train']['batch_size']),
        epochs=config['train']['epochs'],
        callbacks=cbs,
        steps_per_epoch=len(train_X) / config['train']['batch_size'],
        class_weight=class_weights,
        validation_data=(valid_X, valid_y))
    logging.info("Finish training")

    model.save_weights(os.path.join(model_path, exp_name + '.h5'))
    logging.info("Finish saving model")

    # Save the figures
    fig_acc = show_train_history(history, 'acc', 'val_acc')
    plt.savefig(os.path.join(result_path, exp_name + '_acc.png'))

    fig_los = show_train_history(history, 'loss', 'val_loss')
    plt.savefig(os.path.join(result_path, exp_name + '_loss.png'))

    print("finish!")
