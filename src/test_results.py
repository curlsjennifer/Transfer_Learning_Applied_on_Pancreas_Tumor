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
                                     get_patches, get_patches_list, load_patches, convert_csv_to_dict)
from models.net_keras import *
from data_description.visualization import plot_roc, show_train_history

# Load Configs
def test_model(config, exp_name, mode='mix', fold=5):

    # Env Settings
    os.environ["CUDA_VISIBLE_DEVICES"] = config['system']['CUDA_VISIBLE_DEVICES']
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    set_session(tf.Session(config=sess_config))

    exp_path = os.path.join(config['log']['result_dir'], exp_name)
    roc_path = exp_path + '/rocs/'
    exp_names = [file.split('.')[0] for file in os.listdir( exp_path + '/models/')]
    if os.path.isdir(roc_path):
        rmtree(roc_path)
    os.makedirs(roc_path)


    # Work on each data
    index = 0
    print('Processing on experiment ', exp_name)
    for filename in exp_names:
        res_list = exp_res(config, filename, 'ntuh')
        if index > 0:
            Result_ntuh = pd.concat([Result_ntuh, res_list])
        else:
            Result_ntuh = res_list
        index += 1

    index = 0
    for filename in exp_names:
        res_list = exp_res(config, filename, 'ext')
        if index > 0:
            Result_ext = pd.concat([Result_ext, res_list])
        else:
            Result_ext = res_list
        index += 1

    Result_ntuh.to_csv(os.path.join(exp_path, exp_name + '_ntuh.csv'))
    Result_ext.to_csv(os.path.join(exp_path, exp_name + '_ext.csv'))

    ntuh_list = Result_ntuh['AUC'].values.tolist()
    ext_list = Result_ext['AUC'].values.tolist()
    data = pd.DataFrame({
        'title':['mean', 'std', 'min', 'max'],
        'ntuh':[np.mean(ntuh_list),np.std(ntuh_list),min(ntuh_list), max(ntuh_list)],
        'ext':[np.mean(ext_list),np.std(ext_list),min(ext_list), max(ext_list)]})
    data.to_csv(os.path.join(exp_path, exp_name + '_info.csv'))

    # x_label = range(fold)
    # plt.figure()
    # plt.title('AUC')
    # plt.plot(x_label, Result_ntuh['AUC'].values.tolist())
    # plt.plot(x_label, Result_ext['AUC'].values.tolist())
    # plt.legend(['ntuh', 'ext'], loc='lower right')
    # plt.xlim([min(x_label), max(x_label)])
    # plt.ylim([0, 1])
    # plt.ylabel('AUC')
    # plt.show()
    # plt.savefig(os.path.join(exp_path, exp_name + '.png'))

def exp_res(config, exp_name, type_ext):
    print('Processing on ', exp_name)
    exp_path = os.path.join(config['log']['result_dir'], '_'.join(exp_name.split('_')[:-1]))
    with open(os.path.join(exp_path,  'jsons', exp_name + '.json') , 'r') as reader:
        case = json.loads(reader.read())
        case_partition = case['partition']

    # Load validation set
    valid_X, valid_y, valid_idx = get_patches_list(config, case_partition, mode='validation')
    valid_X = np.array(valid_X)
    valid_X = valid_X.reshape(valid_X.shape[0], valid_X.shape[1], valid_X.shape[2], 1)
    valid_y = np.array(valid_y)

    # Load test set
    if type_ext == 'ext':
        case_partition[0]['test'] = []
    else:
        case_partition[1]['test'] = []
        case_partition[2]['test'] = []

    test_X, test_y, test_idx = get_patches_list(config, case_partition, mode='test')
    test_X = np.array(test_X)
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], test_X.shape[2], 1)
    test_y = np.array(test_y)


    # Build the model
    model = eval(config['model']['name'])(config['dataset']['input_dim'])

    model.load_weights(exp_path + '/models/' + exp_name + '.h5')
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
    plt.savefig(os.path.join(exp_path, 'rocs', exp_name + '_' + type_ext + '.png'))

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
    return pd.DataFrame({'exp_name': exp_name, 'AUC': [roc_auc],'accuracy': [accuracy],'sensitivity': [sensitivity],'specificity': [specificity], 
                        'TP': [patch_matrix[0][0]], 'FP': [patch_matrix[1][0]], 'FN': [patch_matrix[0][1]], 'PN': [patch_matrix[1][1]]})

