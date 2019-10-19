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

import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, roc_curve, auc

from utils import load_config, flatten_config_for_logging
from data_loader.data_loader import (DataGenerator_NTUH,
                                     load_patches, split_save_case_partition,
                                     load_list)
from models.net_keras_svd import *
from data_description.visualization import plot_roc, show_train_history

plt.switch_backend('agg')

# Parse Args
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default='./configs/base.yml',
                    type=str, help="train configuration")
parser.add_argument("-r", "--run_name", default=None, type=str,
                    help="run name for this experiment. (Default: time)")
parser.add_argument("-comet", "--comet_implement", default=False,
                    type=bool, help="record logging in comet")
args = parser.parse_args()

# Load config
config = load_config(args.config)
if args.run_name is None:
    config['run_name'] = strftime("%Y%m%d_%H%M%S", localtime())
else:
    config['run_name'] = args.run_name
pprint(config)

# Set logging
logging_filename = os.path.join(
    config['log']['checkpoint_dir'], config['run_name'] + '_file.log')
logging.root.removeHandler(absl_log._absl_handler)
absl_log._warn_preinit_stderr = False
logging.basicConfig(
    filename=logging_filename,
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format='%(levelname)-8s: %(asctime)-12s: %(message)s'
)

# Set path
model_path = os.path.join(config['log']['model_dir'], config['run_name'])
if not os.path.isdir(model_path):
    os.mkdir(model_path)
result_basepath = os.path.join(
    config['log']['result_dir'], config['run_name'])
if not os.path.isdir(result_basepath):
    os.mkdir(result_basepath)

# Record experiment in Comet.ml
if args.comet_implement:
    experiment = Experiment(api_key=config['comet']['api_key'],
                            project_name=config['comet']['project_name'],
                            workspace=config['comet']['workspace'])
    experiment.logging_parameters(flatten_config_for_logging(config))
    experiment.add_tags(
        [config['model']['name'], config['dataset']['dir'].split('/')[-1]])
    experiment.logging_asset(file_path=args.config)


# Env settings
os.environ["CUDA_VISIBLE_DEVICES"] = config['system']['CUDA_VISIBLE_DEVICES']
gpu_options = tf.GPUOptions(allow_growth=True)
sess_config = tf.ConfigProto(gpu_options=gpu_options)
set_session(tf.Session(config=sess_config))

# split cases into train, val, test
data_list = load_list(config['dataset']['csv'])

healthy_partition = split_save_case_partition(data_list['healthy_train'], ratio=(
    0.8, 0.2, 0), test_cases=data_list['healthy_test'], random_seed=1)
tumor_partition = split_save_case_partition(data_list['tumor_train'], ratio=(
    0.8, 0.2, 0), test_cases=data_list['tumor_test'], random_seed=1)

case_partition = {}
case_partition['validation'] = healthy_partition['validation'] + \
    tumor_partition['validation']
case_partition['train'] = healthy_partition['train'] + tumor_partition['train']
case_partition['test'] = healthy_partition['test'] + tumor_partition['test']
logging.info("Case partition: {}".format(case_partition))

# For test - to check if the whole code work
# case_partition['train'] = case_partition['train'][-5:]
# case_partition['validation'] = case_partition['validation'][-5:]
# case_partition['test'] = case_partition['test'][-5:]

# Get patches
patch_size = config['dataset']['input_dim'][0]
stride = config['dataset']['stride']
DataGenerator_train = DataGenerator_NTUH(
    config['dataset']['dir'], patch_size, stride=stride)
train_X = []
train_y = []
for case_id in tqdm(case_partition['train']):
    box_img, box_pan, box_les = DataGenerator_train.load_boxdata(case_id)
    image, pancreas, lesion = DataGenerator_train.preprocessing(
        box_img, box_pan, box_les)
    tmp_X, tmp_y = DataGenerator_train.generate_patch(
        image, pancreas, lesion)
    train_X = train_X + tmp_X
    train_y = train_y + tmp_y

train_X = np.array(train_X)
train_X = train_X.reshape(
    train_X.shape[0], train_X.shape[1], train_X.shape[2], 1)
train_y = np.array(train_y)
print("Finish loading {} patches from {} studies".format(
    train_X.shape[0], len(case_partition['train'])))
logging.info("Finish loading {} patches from {} studies".format(
    train_X.shape[0], len(case_partition['train'])))
print("With {} lesion patches and {} normal pancreas patches".format(
    np.sum(train_y), train_X.shape[0] - np.sum(train_y)))
logging.info("With {} lesion patches and {} normal pancreas patches".format(
    np.sum(train_y), train_X.shape[0] - np.sum(train_y)))

DataGenerator_valid = DataGenerator_NTUH(config['dataset']['dir'], patch_size)
valid_X = []
valid_y = []
for case_id in tqdm(case_partition['validation']):
    box_img, box_pan, box_les = DataGenerator_valid.load_boxdata(case_id)
    image, pancreas, lesion = DataGenerator_valid.preprocessing(
        box_img, box_pan, box_les)
    tmp_X, tmp_y = DataGenerator_valid.generate_patch(
        image, pancreas, lesion)
    valid_X = valid_X + tmp_X
    valid_y = valid_y + tmp_y

valid_X = np.array(valid_X)
valid_X = valid_X.reshape(
    valid_X.shape[0], valid_X.shape[1], valid_X.shape[2], 1)
valid_y = np.array(valid_y)
print("Finish loading {} patches from {} studies".format(
    valid_X.shape[0], len(case_partition['validation'])))
logging.info("Finish loading {} patches from {} studies".format(
    valid_X.shape[0], len(case_partition['validation'])))
print("With {} lesion patches and {} normal pancreas patches".format(
    np.sum(valid_y), valid_X.shape[0] - np.sum(valid_y)))
logging.info("With {} lesion patches and {} normal pancreas patches".format(
    np.sum(valid_y), valid_X.shape[0] - np.sum(valid_y)))

# Data Generators - Keras
datagen = ImageDataGenerator(
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

model.save_weights(os.path.join(model_path, 'weights.h5'))
logging.info("Finish savining model")

logging.info("Start running test")
DataGenerator_test = DataGenerator_NTUH(config['dataset']['dir'], patch_size)
test_X = []
test_y = []
prediction_chart = pd.DataFrame(columns=['case_id', 'tp', 'fn', 'fn', 'tn'])
for index, case_id in tqdm(enumerate(case_partition['test'])):
    box_img, box_pan, box_les = DataGenerator_test.load_boxdata(case_id)
    image, pancreas, lesion = DataGenerator_test.preprocessing(
        box_img, box_pan, box_les)
    tmp_X, tmp_y = DataGenerator_test.generate_patch(
        image, pancreas, lesion)
    test_X = test_X + tmp_X
    test_y = test_y + tmp_y

    DataGenerator_test.get_prediction(model)
    tp, fn, fp, tn = DataGenerator_test.get_all_value()
    prediction_chart.loc[index] = [case_id, tp, fn, fp, tn]


test_X = np.array(test_X)
test_X = test_X.reshape(
    test_X.shape[0], test_X.shape[1], test_X.shape[2], 1)
test_y = np.array(test_y)
print("Finish loading {} patches from {} studies".format(
    test_X.shape[0], len(case_partition['test'])))
logging.info("Finish loading {} patches from {} studies".format(
    test_X.shape[0], len(case_partition['test'])))
print("With {} lesion patches and {} normal pancreas patches".format(
    np.sum(test_y), test_X.shape[0] - np.sum(test_y)))
logging.info("With {} lesion patches and {} normal pancreas patches".format(
    np.sum(test_y), test_X.shape[0] - np.sum(test_y)))

loss, accuracy = model.evaluate(test_X, test_y)
print('loss = ', loss, 'accuracy = ', accuracy)


# Save the result
fig_acc = show_train_history(history, 'acc', 'val_acc')
plt.savefig(os.path.join(result_basepath, 'acc_plot.png'))

fig_los = show_train_history(history, 'loss', 'val_loss')
plt.savefig(os.path.join(result_basepath, 'loss_plot.png'))

model.save_weights(os.path.join(model_path, 'weights.h5'))
model.save(os.path.join(model_path, 'model.h5'))
