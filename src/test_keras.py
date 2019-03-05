import os
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
from sklearn.metrics import confusion_matrix, roc_curve

from utils import load_config, flatten_config_for_logging
from data_loader.data_loader import (convert_csv_to_dict,
                                     load_patches)
from models.net_keras import *
from data_description.visualization import plot_roc
from data_loader.patch_sampler import patch_generator

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

# Load test data and make prediction
patch_size = config['dataset']['input_dim'][0]
test_X, test_y = load_patches(
    config['dataset']['dir'], case_partition['test'], patch_size=patch_size,
    train_mode=False)

loss, accuracy = model.evaluate(test_X, test_y)
print('loss = ', loss, 'accuracy = ', accuracy)

y_predict = model.predict_classes(test_X)
print('Ground true tumor:', np.sum(test_y),
      'Ground true pancreas:', test_y.shape[0]-np.sum(test_y))
print('Predicted tumor:', np.sum(y_predict),
      'Predicted pancreas:', y_predict.shape[0]-np.sum(y_predict))
print(confusion_matrix(test_y, y_predict, labels=[1, 0]))

probs = model.predict_proba(test_X)
patch_fig = plot_roc(probs, test_y)
plt.savefig(os.path.join(result_path, 'patch_roc.png'))

patch_fpr, patch_tpr, patch_thresholds = roc_curve(test_y, probs)

index_start = np.argmax(1-patch_fpr + patch_tpr)
print(patch_tpr[index_start])
threshold = patch_thresholds[index_start]

tpr_val = np.ceil(patch_tpr[index_start]*10)/10
while tpr_val <= 1:
    index_roc = (np.abs(patch_tpr - tpr_val)).argmin()
    threshold = patch_thresholds[index_roc]
    print('TP:', tpr_val, 'threshold:', threshold)

    tpr_val = tpr_val + 0.05

    # Patient-based
    patient_prediction = pd.DataFrame(
        columns=['case_id', 'detected_patches', 'total_patches', 'prediction'])
    patient_y = []
    for index, case_id in enumerate(case_partition['test']):
        if case_id[:2] == 'AD' or case_id[:2] == 'NP':
            patient_y.append(0)
            X, y = patch_generator(
                config['dataset']['dir'], case_id, patch_size,
                max_amount=100, train_mode=False)
            test_X = np.array(X)
            test_X = test_X.reshape(
                (test_X.shape[0], test_X.shape[1], test_X.shape[2], 1))
            y_predict = model.predict_proba(test_X)
            y_predict[y_predict < threshold] = 0
            y_predict[y_predict > threshold] = 1
            patient_prediction.loc[index] = [case_id, np.sum(
                y_predict), len(y_predict), np.mean(y_predict)]
        else:
            patient_y.append(1)
            X, y = patch_generator(
                config['dataset']['dir'], case_id, patch_size,
                max_amount=50, train_mode=False)
            test_X = np.array(X)
            test_X = test_X.reshape(
                (test_X.shape[0], test_X.shape[1], test_X.shape[2], 1))
            y_predict = model.predict_proba(test_X)
            y_predict[y_predict < threshold] = 0
            y_predict[y_predict >= threshold] = 1
            patient_prediction.loc[index] = [case_id, np.sum(
                y_predict), len(y_predict), np.mean(y_predict)]
    tpr_name = str(np.floor((tpr_val-0.05)*100)/100)
    patient_prediction.to_csv(
        os.path.join(result_path, ('patient_prediction_' + tpr_name + '.csv')))
    fpr, tpr, thresholds = roc_curve(
        patient_y, list(patient_prediction['prediction']))

    patient_fig = plot_roc(list(patient_prediction['prediction']), patient_y)
    plt.savefig(os.path.join(result_path, ('patient_roc' + tpr_name + '.png')))

print('ALL DONE!')
