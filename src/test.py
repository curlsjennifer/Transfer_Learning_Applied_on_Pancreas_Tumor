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
import collections
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
from data_loader.data_loader import get_patches, dataset, exp_path
from net_keras import *


# Work on each data
def exp_res(filename, config, type_target):
    
    # source and target data generation
    name = exp_path(filename, inc=True)
    if type_target == 'source':
        [train, valid, test] = np.load(
            name.source_path, allow_pickle=True)
    else:
        [train, valid, test] = np.load(
            name.target_path, allow_pickle=True)

    valid_X = valid.X
    valid_y = valid.y
    test_X = test.X
    test_y = test.y

    # Load and complie model
    model = eval(config['model']['name'])(config['dataset']['input_dim'])
    model.load_weights(name.model_path)

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
    if type_target == 'source':
        plt.savefig(name.roc_source_path)
    else:
        plt.savefig(name.roc_target_path)

    # Calculate other data
    patch_fpr, patch_tpr, patch_thresholds = roc_curve(test_y, probs)
    roc_auc = auc(patch_fpr, patch_tpr)

    loss, accuracy = model.evaluate(test_X, test_y)
    y_predict = model.predict_classes(test_X)

    probs_binary = predict_binary(probs, patch_threshold)
    # false_item = np.reshape(test_y, (len(probs), 1))
    #             - np.reshape(probs, (len(probs), 1))
    
    # idx = []
    # for it_source, it_name, it_len in test.idx:
    #     idx.extend([it_name] * it_len)
    # false_item_test = [idx[i] for i in range(len(idx)) if false_item[i]==1]
    # false_item_true = [idx[i] for i in range(len(idx)) if false_item[i]==-1]
    
    # coll_test = collections.Counter(false_item_test)
    # coll_true = collections.Counter(false_item_true)
    
    # name_false = pd.DataFrame({'exp_name': [filename]*len(test.idx),
    #                         'item_source': [test.idx[i][0] for i in range(len(test.idx))],
    #                         'item_name': [test.idx[i][1] for i in range(len(test.idx))],
    #                        'false_test': [coll[test.idx[i][1]] for i in range(len(test.idx))],
    #                        'len':[test.idx[i][2] for i in range(len(test.idx))],
    #                        'test_ind':[filename.split('_')[3] for i in range(len(test.idx))],
    #                        'test_num':[filename.split('_')[4] for i in range(len(test.idx))],
    #                        'rate': [coll[test.idx[i][1]]/test.idx[i][2] for i in range(len(test.idx))],
    #                         'AUC':[roc_auc for i in range(len(test.idx))]
    #                        })
    
    patch_matrix = confusion_matrix(test_y, probs_binary, labels=[1, 0])
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
    name_false = []
    
    if type_target == 'source':
        np.save(name.patch_source_path, 
                [test_y, probs_binary, probs, patch_threshold, roc_auc])
    else:
        np.save(name.patch_target_path, 
                [test_y, probs_binary, probs, patch_threshold, roc_auc])
        
    return result, name_false


parser = argparse.ArgumentParser()
parser.add_argument("-r", "--run_name", default=None, type=str,
                    help="run name for this experiment. (Default: time)")
parser.add_argument("-d", "--dev", default=None, type=str,
                    help="device")
args = parser.parse_args()
run_name = args.run_name

# Load Configs
config = load_config('./configs/basic.yml')

# Env Settings
os.environ["CUDA_VISIBLE_DEVICES"] = args.dev
gpu_options = tf.GPUOptions(allow_growth=True)
sess_config = tf.ConfigProto(gpu_options=gpu_options)
set_session(tf.Session(config=sess_config))

# Work on target test set
exp_list = os.listdir(''.join(['../results/', run_name, '/models/']))
exp_list = [exp.replace('weights', run_name).replace('.h5', '') for exp in exp_list]

index = 0
for filename in exp_list:
    print(filename)
    res_list, name_false = exp_res(filename, config, 'source')
    Result_source = pd.concat([Result_source, res_list]) if index == 1 else res_list
    #False_source = pd.concat([False_source, name_false]) if index == 1 else name_false
    index = 1
    
Result_source.to_csv(''.join([
    '../results/', run_name, '/source.csv']))
# False_source.to_csv(''.join([
#     '../results/', run_name, '/source_false.csv']))

index = 0
for filename in exp_list:
    print(filename)
    res_list, name_false = exp_res(filename, config, 'target')
    Result_target = pd.concat([Result_target, res_list]) if index == 1 else res_list
    #False_target = pd.concat([False_target, name_false]) if index == 1 else name_false
    index = 1

Result_target.to_csv(''.join([
    '../results/', run_name, '/target.csv']))
# False_target.to_csv(''.join([
#     '../results/', run_name, '/target_false.csv']))