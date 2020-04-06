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
from shutil import copyfile, rmtree, copytree
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
                                     get_patches, get_patches_test, load_patches, convert_csv_to_dict, get_patches_list)
from models.net_keras import *
from data_description.visualization import plot_roc, show_train_history



# Work on each data
def exp_res(filename, config, type_ext, trans):
    print('Processing on ', filename)

    with open(filename , 'r') as reader:
        case = json.loads(reader.read())
        case_partition = case['partition']

    valid_X, valid_y, valid_idx = get_patches(
        config, case_partition, mode='validation')

    valid_X = np.array(valid_X)
    valid_X = valid_X.reshape(
        valid_X.shape[0], valid_X.shape[1], valid_X.shape[2], 1)
    valid_y = np.array(valid_y)

    # ntuh and ext data generation
    i = 0

    if type_ext == 'ntuh':
        test_splits = [80, 80, 0, 0]
    else:
        test_splits = [0, 0, 17, 56]

    case_partition = case['partition']

    test_X, test_y, test_idx = get_patches_test(
        config, [case_partition[0], case_partition[1], case_partition[2]], test_split=test_splits)

    test_X = np.array(test_X)
    test_X = test_X.reshape(
        test_X.shape[0], test_X.shape[1], test_X.shape[2], 1)
    test_y = np.array(test_y)

    model = eval(config['model']['name'])(config['dataset']['input_dim'])

    if trans == "trans":
        model.load_weights(os.path.join(
            config['log']['model_dir'], filename.split('/')[-1].split('.')[0] + '_trans', 'weights.h5'))
        weights, biases = model.layers[0].get_weights()
    else:
        model.load_weights(os.path.join(
            config['log']['model_dir'], filename.split('/')[-1].split('.')[0], 'weights.h5'))
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
    if trans == "trans":
        plt.savefig(os.path.join(exp_path + '/rocs/' , 'roc_' + type_ext + '_' + filename.split('/')[-1] + '_trans.png'))
    else:
        plt.savefig(os.path.join(exp_path + '/rocs/' , 'roc_' + type_ext + '_' + filename.split('/')[-1] + '.png'))

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


parser = argparse.ArgumentParser()
parser.add_argument("-r", "--run_name", default=None, type=str,
                    help="run name for this experiment. (Default: time)")
parser.add_argument("-t", "--trans", default=None, type=str,
                    help="trans")                   
args = parser.parse_args()

# Load Configs
config = load_config('./configs/test_result.yml') 

# Env Settings
os.environ["CUDA_VISIBLE_DEVICES"] = config['system']['CUDA_VISIBLE_DEVICES']
gpu_options = tf.GPUOptions(allow_growth=True)
sess_config = tf.ConfigProto(gpu_options=gpu_options)
set_session(tf.Session(config=sess_config))

# Create Experiment Folder
exp_path = os.path.join(config['log']['result_dir'], args.run_name)
if not os.path.isdir(exp_path + '/models/'):
    os.mkdir(exp_path + '/models/')
    os.mkdir(exp_path + '/figures/')
    os.mkdir(exp_path + '/rocs/')
    os.mkdir(exp_path + '/trans_models/')
    os.mkdir(exp_path + '/trans_figures/')
    os.mkdir(exp_path + '/trans_rocs/')

index = 0
print('Processing on experiment ', args.run_name)
for filename in os.listdir(exp_path + '/jsons/'):
    name = filename.split('.')[0]
    res_list = exp_res(exp_path + '/jsons/' + filename, config, 'ntuh', args.trans)
    if index > 0:
        Result_ntuh = pd.concat([Result_ntuh, res_list])
    else:
        Result_ntuh = res_list
        index += 1


index = 0
for filename in os.listdir(exp_path + '/jsons/'):
    if args.trans == "trans":
        name = filename.split('.')[0] + "_trans"
    else:
        name = filename.split('.')[0]
    res_list = exp_res(exp_path + '/jsons/' + filename, config, 'ext', args.trans)
    if index > 0:
            Result_ext = pd.concat([Result_ext, res_list])
    else:
        Result_ext = res_list
        index += 1

    copyfile(os.path.join(config['log']['result_dir'], name,'acc_plot.png'), exp_path + '/figures/acc_' + name + '.png')
    copyfile(os.path.join(config['log']['result_dir'], name,'loss_plot.png'), exp_path + '/figures/loss_' + name + '.png')
    copyfile(os.path.join(config['log']['model_dir'], name , 'weights.h5'), exp_path + '/models/' + name + '.h5')

if args.trans == "trans":    
    Result_ntuh.to_csv(os.path.join(exp_path, (args.run_name + '_ntuh_trans.csv')))
    Result_ext.to_csv(os.path.join(exp_path, (args.run_name + '_ext_trans.csv')))
else:
    Result_ntuh.to_csv(os.path.join(exp_path, (args.run_name + '_ntuh.csv')))
    Result_ext.to_csv(os.path.join(exp_path, (args.run_name + '_ext.csv')))


