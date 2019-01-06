from comet_ml import Experiment
import os
from random import shuffle
from pprint import pprint
from ast import literal_eval
from tqdm import trange, tqdm
import numpy as np

import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from sklearn.metrics import f1_score, classification_report

from models.net_keras import simple_cnn_keras
from data_loader.data_loader import fix_save_case_partition, load_patches
from utils import get_config_sha1, f1_keras_metric


# Load config
with open('./configs/config_tinghui.txt', 'r') as f:
    config = literal_eval(f.read())
    config['config_sha1'] = get_config_sha1(config, 5)
    pprint(config)

# Env settings
os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_VISIBLE_DEVICES']
# Session config, limiting gpu memory - Keras
# sess_config = tf.ConfigProto()
# sess_config.gpu_options.per_process_gpu_memory_fraction = config['GPU_memory_fraction']

gpu_options = tf.GPUOptions(allow_growth=True)
sess_config = tf.ConfigProto(gpu_options=gpu_options)

set_session(tf.Session(config=sess_config))

# Record experiment in Comet.ml
experiment = Experiment(api_key="vxjwm9gKFuiJaYpfHDBto3EgN",
                        project_name="general", workspace="tinghui")
experiment.log_parameters(config)
experiment.add_tag('keras')
experiment.add_tag(config['model'])

# split cases into train, val, test
case_list = os.listdir(config['case_list_dir'])
case_partition = fix_save_case_partition(
    case_list, config['case_split_ratio'], path=config['case_partition_path'],
    random_seed=config['random_seed']) if config['case_partition_path'] == '' else load_case_partition(config['case_partition_path'])

# # Get patch partition
# patch_partition, patch_paths, labels = get_patch_partition_labels(
#     case_partition, config['patch_pancreas_dir'], config['patch_lesion_dir'])

# Get patches
train_X, train_y = load_patches(config['case_list_dir'],
                                case_partition['train'])
print("Finish loading {} patches from {} studies".format(train_X.shape[0], len(case_partition['train'])))
valid_X, valid_y = load_patches(config['case_list_dir'],
                                case_partition['validation'])
print("Finish loading {} patches from {} studies".format(valid_X.shape[0], len(case_partition['validation'])))

# Shuffle data
shuffle_indices = np.random.permutation(np.arange(len(train_y)))
train_X = train_X[shuffle_indices]
train_y = train_y[shuffle_indices]

train_y = keras.utils.to_categorical(train_y, num_classes=2)
valid_y = keras.utils.to_categorical(valid_y, num_classes=2)


# # Data Generators - Keras
# training_generator = DataGenerator_keras(
#     train_X, train_y,
#     case_partition['train'], labels, patch_paths,
#     config['batch_size'], config['input_dim'][:2], config['input_dim'][2], 2, True)
# validation_generator = DataGenerator_keras(
#     valid_X, valid_y,
#     case_partition['validation'], labels, patch_paths,
#     config['batch_size'], config['input_dim'][:2], config['input_dim'][2], 2, False)

# Model Init
model = simple_cnn_keras(config['input_dim'], num_classes=2)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=config['lr'], amsgrad=True),
              metrics=['accuracy', f1_keras_metric])
cbs = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1)]

# # Model fitting
# model.fit_generator(generator=training_generator,
#                     validation_data=validation_generator,
#                     epochs=config['epochs'],
#                     verbose=1,
#                     callbacks=cbs,
#                     use_multiprocessing=True,
#                     workers=config['num_cpu'])

model.fit(train_X,
          train_y,
          epochs=config['epochs'],
          callbacks=cbs,
          batch_size=32,
          class_weight='auto',
          validation_data=(valid_X, valid_y))
