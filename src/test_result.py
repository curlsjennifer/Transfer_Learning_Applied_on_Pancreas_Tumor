import os, time
from pprint import pprint
import copy
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
from shutil import copyfile, rmtree
# from comet_ml import Experiment

import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn import metrics

from utils import load_config, flatten_config_for_logging, find_threshold, predict_binary
from data_loader.data_loader import (DataGenerator, generate_case_partition,
                                     get_patches, get_patches_test, load_patches, convert_csv_to_dict)
from models.net_keras import *
from data_description.visualization import plot_roc, show_train_history

# Load Configs
config = load_config('./configs/test_result.yml') 

# Env Settings
os.environ["CUDA_VISIBLE_DEVICES"] = config['system']['CUDA_VISIBLE_DEVICES']
gpu_options = tf.GPUOptions(allow_growth=True)
sess_config = tf.ConfigProto(gpu_options=gpu_options)
set_session(tf.Session(config=sess_config))

# Create Experiment Folder
exp_path = os.path.join(config['log']['result_dir'], config['experiment']['foldername'])
if os.path.isdir(exp_path):
    rmtree(exp_path)
os.mkdir(exp_path)
os.mkdir(exp_path + '/json/')
os.mkdir(exp_path + '/model/')

first_folder = config['experiment']['folders'][0]
json_name = os.path.join(config['log']['checkpoint_dir'],  first_folder + '_info.json')

# Load case partition
with open(json_name , 'r') as reader:
    case = json.loads(reader.read())
    case_partition = case['partition']
    dataset_information = case["dataset_information"]
    print("NTUH_split_H", dataset_information['NTUH_split_H'])
    print("NTUH_split_C", dataset_information['NTUH_split_C'])
    print("PancreasCT_split", dataset_information['PancreasCT_split'])
    print("MSD_split", dataset_information['MSD_split'])

# Load validation and test data
valid_list = case_partition[0]['validation'] + case_partition[1]['validation'] + case_partition[2]['validation']
test_list_ntuh =  case_partition[0]['test']
test_list_ext =  case_partition[1]['test'] + case_partition[2]['test']

valid_X, valid_y, valid_idx = get_patches(
    config, [case_partition[0], case_partition[1], case_partition[2]], mode='validation')

valid_X = np.array(valid_X)
valid_X = valid_X.reshape(
    valid_X.shape[0], valid_X.shape[1], valid_X.shape[2], 1)
valid_y = np.array(valid_y)

# ntuh and ext data generation
i = 0
test_splits_all = [[80, 80, 0, 0], [0, 0, 17, 56]]
for test_splits in test_splits_all:
    case_partition = case['partition']

    test_X, test_y, test_idx = get_patches_test(
        config, [case_partition[0], case_partition[1], case_partition[2]], test_split=test_splits)

    test_X = np.array(test_X)
    test_X = test_X.reshape(
        test_X.shape[0], test_X.shape[1], test_X.shape[2], 1)
    test_y = np.array(test_y)
    if i == 0:
        Test_X_ntuh = test_X
        Test_y_ntuh = test_y
    else:
        Test_X_ext = test_X
        Test_y_ext = test_y
    test_X = None
    test_y = None
    i = 1

# Work on each data
def exp_res(filename, config, test_X, test_y, valid_X, valid_y, type_ext):
    print('Processing on ', filename)

    # Build the model
    model = eval(config['model']['name'])(config['dataset']['input_dim'])

    model.load_weights(os.path.join(
        config['log']['model_dir'], filename, 'weights.h5'))
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
    plt.savefig(os.path.join(exp_path, 'roc_' + type_ext + '_' + filename + '.png'))

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
    #return pd.DataFrame({'exp_name': "A", 'AUC': [1],'accuracy': [2],'sensitivity': [3],'specificity': [4], 
    #                     'TP': [5], 'FP': [6], 'FN': [7], 'PN': [8]})
    return pd.DataFrame({'exp_name': filename, 'AUC': [roc_auc],'accuracy': [accuracy],'sensitivity': [sensitivity],'specificity': [specificity], 
                        'TP': [patch_matrix[0][0]], 'FP': [patch_matrix[1][0]], 'FN': [patch_matrix[0][1]], 'PN': [patch_matrix[1][1]]})

index = 0
print('Processing on experiment ', config['experiment']['foldername'])
for filename in config['experiment']['folders']:
    res_list = exp_res(filename, config, Test_X_ntuh, Test_y_ntuh, valid_X, valid_y, 'ntuh')
    if index > 0:
        Result_ntuh = pd.concat([Result_ntuh, res_list])
    else:
        Result_ntuh = res_list
    # copyfile(os.path.join(config['log']['result_dir'], filename,'acc_plot.png'), exp_path + '/acc_' + filename + '.png')
    # copyfile(os.path.join(config['log']['result_dir'], filename,'loss_plot.png'), exp_path + '/loss_' + filename + '.png')
    # copyfile(os.path.join(config['log']['checkpoint_dir'], filename + '_info.json'), exp_path + '/json/' + filename + '_info.json')
    # copyfile(os.path.join(config['log']['model_dir'], filename , 'weights.h5'), exp_path + '/models/' + filename + '.h5')
    index += 1

index = 0
for filename in config['experiment']['folders']:
    res_list = exp_res(filename, config, Test_X_ext, Test_y_ext, valid_X, valid_y, 'ext')
    if index > 0:
            Result_ext = pd.concat([Result_ext, res_list])
    else:
        Result_ext = res_list
    index += 1

Result_ntuh.to_csv(os.path.join(exp_path, (config['experiment']['foldername'] + '_ntuh.csv')))
Result_ext.to_csv(os.path.join(exp_path, (config['experiment']['foldername'] + '_ext.csv')))

# if not config['experiment']['x_label']:
#     config['experiment']['x_label'] = range(len(Result_ntuh['AUC']))

# x_label = config['experiment']['x_label']
# plt.figure()
# plt.title('AUC')
# plt.plot(x_label, Result_ntuh['AUC'].values.tolist())
# plt.plot(x_label, Result_ext['AUC'].values.tolist())
# plt.legend(['ntuh', 'ext'], loc='lower right')
# plt.xlim([min(x_label), max(x_label)])
# plt.ylim([0, 1])
# plt.ylabel('AUC')
# plt.show()
# plt.savefig(os.path.join(exp_path, 'AUC_' + config['experiment']['foldername'] + '.png'))
