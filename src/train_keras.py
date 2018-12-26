from comet_ml import Experiment
import os
from random import shuffle
from pprint import pprint
from ast import literal_eval
from tqdm import trange, tqdm

import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from sklearn.metrics import f1_score, classification_report

from models.net import res_2dcnn, pred_to_01, simple_cnn_keras
from models.data_loader import split_save_case_partition, load_case_partition, get_patch_partition_labels, Dataset, DataGenerator_keras
from utils import get_config_sha1, f1_keras_metric


# Load config
with open('./configs/config_keras.txt', 'r') as f:
    config = literal_eval(f.read())
    config['config_sha1'] = get_config_sha1(config, 5)
    pprint(config)

# Env settings
os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_VISIBLE_DEVICES']
# Session config, limiting gpu memory - Keras
sess_config = tf.ConfigProto()
sess_config.gpu_options.per_process_gpu_memory_fraction = config['GPU_memory_fraction']
set_session(tf.Session(config=sess_config))

# Record experiment in Comet.ml
experiment = Experiment(api_key="fdb4jkVkz4zT8vtOYIRIb0XG7",
                        project_name="pancreas-2d", workspace="adamlin120")
experiment.log_parameters(config)
experiment.add_tag('keras')
experiment.add_tag(config['model'])

# split cases into train, val, test
case_list = os.listdir(config['case_list_dir'])
case_partition = split_save_case_partition(
    case_list, config['case_split_ratio'], path=config['case_partition_path'],
    random_seed=config['random_seed']) if config['case_partition_path'] == '' else load_case_partition(config['case_partition_path'])

# Get patch partition
patch_partition, patch_paths, labels = get_patch_partition_labels(
    case_partition, config['patch_pancreas_dir'], config['patch_lesion_dir'])

# Data Generators - Keras
training_generator = DataGenerator_keras(
    patch_partition['train'], labels, patch_paths,
    config['batch_size'], config['input_dim'][:2], config['input_dim'][2], 2, True)
validation_generator = DataGenerator_keras(
    patch_partition['validation'], labels, patch_paths,
    config['batch_size'], config['input_dim'][:2], config['input_dim'][2], 2, False)

# Model Init
model = simple_cnn_keras(config['input_dim'][:2], num_classes=2)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=config['lr'], amsgrad=True),
              metrics=['accuracy', f1_keras_metric])
cbs = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1)]

# Model fitting
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=config['epochs'],
                    verbose=1,
                    callbacks=cbs,
                    use_multiprocessing=True,
                    workers=config['num_cpu'])
