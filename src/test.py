"""
filename : test.py
author : WanYun, Yang
date : 2020/04/12
description :
    calculate the AUC of the cross validation experiments.
"""

import os
import copy
import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from shutil import copyfile, rmtree, copytree

import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn import metrics

from utils import load_config, find_threshold, predict_binary
from data_loader.data_loader import get_patches
from net_keras import *


# Work on each data
def exp_res(filename, config, type_ext, trans):
    print('Processing on ', filename)

    with open(filename , 'r') as reader:
        case = json.loads(reader.read())
        case_partition = case['partition']

    valid_X, valid_y, valid_idx = get_patches(
        config, case_partition, mode='validation')

    valid_X = np.array(valid_X)
    valid_X = np.expand_dims(valid_X, axis=-1)
    valid_y = np.array(valid_y)

    # ntuh and ext data generation
    if type_ext == 'ntuh':
        case_partition[1]['test'] = []
        case_partition[2]['test'] = []
    else:
        case_partition[0]['test'] = []

    test_X, test_y, test_idx = get_patches(
        config, case_partition, mode='test')

    test_X = np.array(test_X)
    test_X = np.expand_dims(test_X, axis=-1)
    test_y = np.array(test_y)

    # Load and complie model
    model = eval(config['model']['name'])(config['dataset']['input_dim'])
    model_name = filename.split('/')[-1].split('.')[0]
    if trans == "trans":
        model.load_weights(os.path.join(
            '../models', model_name + '_trans', 'weights.h5'))
        weights, biases = model.layers[0].get_weights()
    else:
        model.load_weights(os.path.join(
            '../models', model_name, 'weights.h5'))
        weights, biases = model.layers[0].get_weights()

    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.Adam(lr=config['optimizer']['lr'],
                                        amsgrad=True),
        metrics=['accuracy'])
    valid_probs = model.predict_proba(valid_X)
    patch_threshold = find_threshold(valid_probs, valid_y)

    # Test data
    probs = model.predict_proba(test_X)

    # plot roc
    fpr, tpr, threshold = metrics.roc_curve(test_y, probs)
    roc_auc = metrics.auc(fpr, tpr)

    # Draw figures
    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    # Save figures
    fig_name = 'roc_' + type_ext + '_' + filename.split('/')[-1]
    if trans == "trans":
        plt.savefig(os.path.join(exp_path + '/rocs/',
                                 fig_name + '_trans.png'))
    else:
        plt.savefig(os.path.join(exp_path + '/rocs/',
                                 fig_name + '.png'))

    # Calculate other data
    patch_fpr, patch_tpr, patch_thresholds = roc_curve(test_y, probs)
    roc_auc = auc(patch_fpr, patch_tpr)

    loss, accuracy = model.evaluate(test_X, test_y)
    y_predict = model.predict_classes(test_X)

    probs = predict_binary(probs, patch_threshold)
    patch_matrix = confusion_matrix(test_y, probs, labels=[1, 0])
    accuracy = (patch_matrix[0][0] + patch_matrix[1][1]) / test_y.shape[0]
    sensitivity = patch_matrix[0][0] / np.sum(test_y)
    specificity = patch_matrix[1][1] / (test_y.shape[0] - np.sum(test_y))

    print('AUC: ', roc_auc)
    result = pd.DataFrame({'exp_name': filename,
                           'AUC': [roc_auc],
                           'accuracy': [accuracy],
                           'sensitivity': [sensitivity],
                           'specificity': [specificity],
                           'TP': [patch_matrix[0][0]],
                           'FP': [patch_matrix[1][0]],
                           'FN': [patch_matrix[0][1]],
                           'PN': [patch_matrix[1][1]]})
    return result


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp_name", default=None, type=str,
                    help="run name for this experiment. (Default: time)")
parser.add_argument("-t", "--trans", default=None, type=str,
                    help="trans")
args = parser.parse_args()
exp_name = args.exp_name

# Load Configs
config = load_config('./configs/basic.yml')

# Env Settings
os.environ["CUDA_VISIBLE_DEVICES"] = config['system']['CUDA_VISIBLE_DEVICES']
gpu_options = tf.GPUOptions(allow_growth=True)
sess_config = tf.ConfigProto(gpu_options=gpu_options)
set_session(tf.Session(config=sess_config))

# Create Experiment Folder
exp_path = os.path.join('../result', args.exp_name)
if not os.path.isdir(exp_path + '/models/'):
    os.mkdir(exp_path + '/models/')
    os.mkdir(exp_path + '/figures/')
    os.mkdir(exp_path + '/rocs/')
    os.mkdir(exp_path + '/trans_models/')
    os.mkdir(exp_path + '/trans_figures/')
    os.mkdir(exp_path + '/trans_rocs/')

# Work on ntuh test set
index = 0
print('Processing on experiment ', args.exp_name)
for filename in os.listdir(exp_path + '/jsons/'):
    name = filename.split('.')[0]
    res_list = exp_res(exp_path + '/jsons/' + filename,
                       config, 'ntuh', args.trans)
    Result_ntuh = pd.concat([Result_ntuh, res_list]) if index == 1 else res_list
    index = 1

# Work on ext test set
index = 0
for filename in os.listdir(exp_path + '/jsons/'):
    if args.trans == "trans":
        name = filename.split('.')[0] + "_trans"
    else:
        name = filename.split('.')[0]
    res_list = exp_res(exp_path + '/jsons/' + filename,
                       config, 'ext', args.trans)
    Result_ext = pd.concat([Result_ext, res_list]) if index == 1 else res_list

    copyfile(os.path.join('../result', name, 'acc_plot.png'),
             exp_path + '/figures/acc_' + name + '.png')
    copyfile(os.path.join('../result', name, 'loss_plot.png'),
             exp_path + '/figures/loss_' + name + '.png')
    copyfile(os.path.join('../models', name , 'weights.h5'),
             exp_path + '/models/' + name + '.h5')
    index = 1

# Save .csv files
if args.trans == "trans":
    Result_ntuh.to_csv(
        os.path.join(exp_path, (args.exp_name + '_ntuh_trans.csv')))
    Result_ext.to_csv(
        os.path.join(exp_path, (args.exp_name + '_ext_trans.csv')))
else:
    Result_ntuh.to_csv(
        os.path.join(exp_path, (args.exp_name + '_ntuh.csv')))
    Result_ext.to_csv(
        os.path.join(exp_path, (args.exp_name + '_ext.csv')))