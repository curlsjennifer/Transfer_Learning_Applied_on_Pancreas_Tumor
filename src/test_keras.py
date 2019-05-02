import os
import sys
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
from time import gmtime, strftime
import pandas as pd

import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from sklearn.metrics import confusion_matrix, roc_curve, auc

from utils import load_config, flatten_config_for_logging
from data_loader.data_loader import (convert_csv_to_dict,
                                     load_patches)
from models.net_keras import *
from data_description.visualization import plot_roc
from data_loader.patch_sampler import patch_generator

plt.switch_backend('agg')


def predict_binary(prob, threshold):
    binary = np.zeros(prob.shape)
    binary[prob < threshold] = 0
    binary[prob >= threshold] = 1
    return binary


def find_threshold(predict_probs, groundtrue):
    fpr, tpr, thresholds = roc_curve(groundtrue, predict_probs)
    return thresholds[np.argmax(1 - fpr + tpr)]


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
    print('Please add run name!')
else:
    config['run_name'] = args.run_name
pprint(config)

result_path = os.path.join(config['log']['result_dir'], config['run_name'])
if not os.path.isdir(result_path):
    os.mkdir(result_path)

# Env settings
os.environ["CUDA_VISIBLE_DEVICES"] = config['system']['CUDA_VISIBLE_DEVICES']
gpu_options = tf.GPUOptions(allow_growth=True)
sess_config = tf.ConfigProto(gpu_options=gpu_options)
set_session(tf.Session(config=sess_config))

# Load case partition
case_partition = convert_csv_to_dict()

# Build the model
model = eval(config['model']['name'])(config['dataset']['input_dim'])
model.load_weights(os.path.join(
    config['log']['model_dir'], config['run_name'], 'weights.h5'),
    by_name=True)
model.compile(
    loss=keras.losses.binary_crossentropy,
    optimizer=keras.optimizers.Adam(lr=config['optimizer']['lr'],
                                    amsgrad=True),
    metrics=['accuracy'])
print("Finish loading model!")

# Load validation and test data
patch_size = config['dataset']['input_dim'][0]
valid_X, valid_y = load_patches(
    config['dataset']['dir'], case_partition['validation'],
    patch_size=patch_size)
test_X, test_y = load_patches(
    config['dataset']['dir'], case_partition['test'], patch_size=patch_size)

# Find patch-based threshold from validation data
valid_probs = model.predict_proba(valid_X)
patch_threshold = find_threshold(valid_probs, valid_y)

# Find patient-based threshold from validation data
patient_y = []
patient_predict = []
for index, case_id in enumerate(case_partition['validation'] + case_partition['train']):
    if case_id[:2] == 'AD' or case_id[:2] == 'NP':
        patient_y.append(0)
        X, y = patch_generator(
            config['dataset']['dir'], case_id, patch_size,
            stride=5, threshold=0.0004)
        valid_X = np.array(X)
        valid_X = valid_X.reshape(
            (valid_X.shape[0], valid_X.shape[1], valid_X.shape[2], 1))
        y_predict = model.predict_proba(valid_X)
        y_predict = predict_binary(y_predict, patch_threshold)
        patient_predict.append(np.mean(y_predict))
    else:
        patient_y.append(1)
        X, y = patch_generator(
            config['dataset']['dir'], case_id, patch_size,
            stride=5, threshold=0.0004)
        valid_X = np.array(X)
        valid_X = valid_X.reshape(
            (valid_X.shape[0], valid_X.shape[1], valid_X.shape[2], 1))
        y_predict = model.predict_proba(valid_X)
        y_predict = predict_binary(y_predict, patch_threshold)
        patient_predict.append(np.mean(y_predict))
patient_threshold = find_threshold(patient_predict, patient_y)
# print('patient_based threshold:', patient_threshold)

# Test data
probs = model.predict_proba(test_X)
patch_fig = plot_roc(probs, test_y)
plt.savefig(os.path.join(result_path, 'patch_roc.png'))

patch_fpr, patch_tpr, patch_thresholds = roc_curve(test_y, probs)
roc_auc = auc(patch_fpr, patch_tpr)
print('Patch-based AUC:', roc_auc)

loss, accuracy = model.evaluate(test_X, test_y)
print('For patch_based threshold = 0.5:')
print('loss = ', loss, 'accuracy = ', accuracy)

y_predict = model.predict_classes(test_X)
print(confusion_matrix(test_y, y_predict, labels=[1, 0]))

print('patch_based threshold:', patch_threshold)
probs = predict_binary(probs, patch_threshold)
patch_matrix = confusion_matrix(test_y, probs, labels=[1, 0])
print(patch_matrix)
print('accuracy:', (patch_matrix[0][0] + patch_matrix[1][1]) / len(test_y))
print('sensitivity:', patch_matrix[0][0] / np.sum(test_y))
print('specificity:', patch_matrix[1][1] / (test_y.shape[0] - np.sum(test_y)))


# tpr_val = np.ceil(patch_tpr[index_start]*100)/100
# while tpr_val <= 1:

# Patient-based
patient_prediction = pd.DataFrame(
    columns=['case_id', 'detected_patches', 'total_patches', 'prediction'])
patient_y = []
for index, case_id in enumerate(case_partition['test']):
    if case_id[:2] == 'AD' or case_id[:2] == 'NP':
        patient_y.append(0)
        X, y = patch_generator(
            config['dataset']['dir'], case_id, patch_size,
            stride=5, threshold=0.0004)
        test_X = np.array(X)
        test_X = test_X.reshape(
            (test_X.shape[0], test_X.shape[1], test_X.shape[2], 1))
        y_predict = model.predict_proba(test_X)
        y_predict = predict_binary(y_predict, patch_threshold)
        patient_prediction.loc[index] = [case_id, np.sum(
            y_predict), len(y_predict), np.mean(y_predict)]
    else:
        patient_y.append(1)
        X, y = patch_generator(
            config['dataset']['dir'], case_id, patch_size,
            stride=5, threshold=0.0004)
        test_X = np.array(X)
        test_X = test_X.reshape(
            (test_X.shape[0], test_X.shape[1], test_X.shape[2], 1))
        y_predict = model.predict_proba(test_X)
        y_predict = predict_binary(y_predict, patch_threshold)
        patient_prediction.loc[index] = [case_id, np.sum(
            y_predict), len(y_predict), np.mean(y_predict)]

patient_prediction.to_csv(
    os.path.join(result_path, ('patient_prediction.csv')))

fpr, tpr, thresholds = roc_curve(
    patient_y, list(patient_prediction['prediction']))
patient_auc = auc(fpr, tpr)
print('Patient-based AUC:', patient_auc)

patient_fig = plot_roc(list(patient_prediction['prediction']), patient_y)
plt.savefig(os.path.join(result_path, ('patient_roc.png')))

patient_probs = np.array(list(patient_prediction['prediction']))
print('patient_based threshold', patient_threshold)
patient_bin = predict_binary(patient_probs, patient_threshold)
print(confusion_matrix(patient_y, patient_bin, labels=[1, 0]))

patient_new = predict_binary(patient_probs, patient_threshold - 0.05)
print(confusion_matrix(patient_y, patient_new, labels=[1, 0]))

print('ALL DONE!')
