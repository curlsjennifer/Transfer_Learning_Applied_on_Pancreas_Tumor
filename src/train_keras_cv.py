import os
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
from time import gmtime, strftime, localtime
import pandas as pd
import logging

import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, roc_curve, auc

from utils import load_config, flatten_config_for_logging, find_threshold, predict_binary
from data_loader.data_loader import (DataGenerator_NTUH,
                                     load_patches, load_list)
from models.net_keras import *
from data_description.visualization import plot_roc, show_train_history

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
    config['run_name'] = strftime("%Y%m%d_%H%M%S", localtime())
else:
    config['run_name'] = args.run_name
pprint(config)

# Set log
log_filename = os.path.join(
    config['log']['checkpoint_dir'], config['run_name'] + '_file.log')
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format='%(levelname)-8s: %(asctime)-12s: %(message)s'
)

# Set path
model_path = os.path.join(config['log']['model_dir'], config['run_name'])
if not os.path.isdir(model_path):
    os.mkdir(model_path)
result_basepath = os.path.join(config['log']['result_dir'], config['run_name'])
if not os.path.isdir(result_basepath):
    os.mkdir(result_basepath)

# Env settings
os.environ["CUDA_VISIBLE_DEVICES"] = config['system']['CUDA_VISIBLE_DEVICES']
gpu_options = tf.GPUOptions(allow_growth=True)
sess_config = tf.ConfigProto(gpu_options=gpu_options)
set_session(tf.Session(config=sess_config))

data_list = load_list(config['dataset']['csv'])

amount = int(len(data_list['healthy_train']) / 5)

case_list = data_list['healthy_train'] + data_list['tumor_train']
test_list = data_list['healthy_test'] + data_list['tumor_test']


for fold in range(5):
    logging.info("Starting fold {}".format(fold + 1))
    case_partition = {}
    valid_start = fold * amount
    case_partition['validation'] = data_list['healthy_train'][valid_start: valid_start + amount] + \
        data_list['tumor_train'][valid_start: valid_start + amount]
    logging.info("validation list :{}".format(case_partition['validation']))
    case_partition['train'] = list(
        set(case_list).difference(set(case_partition['validation'])))

    result_path = os.path.join(
        result_basepath, 'fold' + str(fold + 1))
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # Get patches
    patch_size = config['dataset']['input_dim'][0]
    stride = config['dataset']['stride']
    DataGenerator_train = DataGenerator_NTUH(
        config['dataset']['dir'], patch_size, stride=stride)
    train_X = []
    train_y = []
    for case_id in tqdm(case_partition['train']):
        box_img, box_pan, box_les = DataGenerator_train.load_boxdata(case_id)
        # ori_img, ori_lbl = DataGenerator_train.load_image(
        #     case_id=case_id, backup_path=config['dataset']['holger'])
        # box_img, box_pan, box_les = DataGenerator_train.get_boxdata(
        #     ori_img, ori_lbl)
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

    DataGenerator_valid = DataGenerator_NTUH(
        config['dataset']['dir'], patch_size)
    valid_X = []
    valid_y = []
    for case_id in case_partition['validation']:
        box_img, box_pan, box_les = DataGenerator_valid.load_boxdata(case_id)
        # ori_img, ori_lbl = DataGenerator_valid.load_image(
        #     case_id=case_id, backup_path=config['dataset']['holger'])
        # box_img, box_pan, box_les = DataGenerator_valid.get_boxdata(
        #     ori_img, ori_lbl)
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
    history = model.fit_generator(
        datagen.flow(train_X, train_y,
                     batch_size=config['train']['batch_size']),
        epochs=config['train']['epochs'],
        callbacks=cbs,
        steps_per_epoch=len(train_X) / config['train']['batch_size'],
        class_weight=class_weights,
        validation_data=(valid_X, valid_y))

    model.save_weights(os.path.join(
        model_path, 'fold' + str(fold + 1) + 'weights.h5'))

    # loss, accuracy = model.evaluate(test_X, test_y)
    # print('loss = ', loss, 'accuracy = ', accuracy)

    # Save the result
    fig_acc = show_train_history(history, 'acc', 'val_acc')
    plt.savefig(os.path.join(result_path, 'acc_plot.png'))

    fig_los = show_train_history(history, 'loss', 'val_loss')
    plt.savefig(os.path.join(result_path, 'loss_plot.png'))

    # Start testing #####################
    logging.info("Start testing")
    # Find patch-based threshold from validation data
    valid_probs = model.predict_proba(valid_X)
    patch_threshold = find_threshold(valid_probs, valid_y)
    logging.info("Patch threshold: {}".format(patch_threshold))

    patch_fpr, patch_tpr, patch_thresholds = roc_curve(valid_y, valid_probs)
    roc_auc = auc(patch_fpr, patch_tpr)
    logging.info('Patch-based AUC: {}'.format(roc_auc))

    probs = predict_binary(valid_probs, patch_threshold)
    patch_matrix = confusion_matrix(valid_y, probs, labels=[1, 0])
    print(patch_matrix)
    logging.info('TP: {}'.format(patch_matrix[0][0]))
    logging.info('FN: {}'.format(patch_matrix[0][1]))
    logging.info('FP: {}'.format(patch_matrix[1][0]))
    logging.info('TN: {}'.format(patch_matrix[1][1]))
    logging.info('accuracy: {}'.format(
        (patch_matrix[0][0] + patch_matrix[1][1]) / len(valid_y)))
    logging.info('sensitivity: {}'.format(patch_matrix[0][0] / np.sum(valid_y)))
    logging.info('specificity: {}'.format(
        patch_matrix[1][1] / (valid_y.shape[0] - np.sum(valid_y))))

    # Find patient-based threshold from validation data
    # patient_y = []
    # patient_predict = []
    # for index, case_id in enumerate(case_partition['validation'] + case_partition['train']):
    #     if case_id[:2] == 'AD' or case_id[:2] == 'NP':
    #         patient_y.append(0)
    #         X, y = patch_generator(
    #             config['dataset']['dir'], case_id, patch_size,
    #             stride=5, threshold=0.0004)
    #         valid_X = np.array(X)
    #         valid_X = valid_X.reshape(
    #             (valid_X.shape[0], valid_X.shape[1], valid_X.shape[2], 1))
    #         y_predict = model.predict_proba(valid_X)
    #         y_predict = predict_binary(y_predict, patch_threshold)
    #         patient_predict.append(np.mean(y_predict))
    #     else:
    #         patient_y.append(1)
    #         X, y = patch_generator(
    #             config['dataset']['dir'], case_id, patch_size,
    #             stride=5, threshold=0.0004)
    #         valid_X = np.array(X)
    #         valid_X = valid_X.reshape(
    #             (valid_X.shape[0], valid_X.shape[1], valid_X.shape[2], 1))
    #         y_predict = model.predict_proba(valid_X)
    #         y_predict = predict_binary(y_predict, patch_threshold)
    #         patient_predict.append(np.mean(y_predict))
    # patient_threshold = find_threshold(patient_predict, patient_y)
    # logging.info('Patient threshold: {}'.format(patient_threshold))

    # for index, case_id in enumerate(test_list):
    #     ori_img, ori_lbl = DataGenerator.load_image(
    #         case_id=case_id, backup_path=holger_path)
    #     box_img, box_pan, box_les = DataGenerator.get_boxdata(ori_img, ori_lbl)
    #     image, pancreas, lesion = DataGenerator.preprocessing(
    #         box_img, box_pan, box_les)
    #     DataGenerator.generate_patch(image, pancreas, lesion)
    #     DataGenerator.get_prediction(model, patch_threshold=patch_threshold)
    #     tp, fn, fp, tn = DataGenerator.get_all_value()
    #     gt_pancreas_patches = DataGenerator.gt_pancreas_num()
    #     print(case_id, fp, gt_pancreas_patches,
    #           fp / gt_pancreas_patches, fn, fp, tn)
    #     info = new_list[new_list['case_id'] == case_id]
    #     if str(info['type'].values[0]) == 'tumor':
    #         gt_lesion_patches = DataGenerator.gt_lesion_num()
    #         patient_prediction.loc[index] = [case_id, 'tumor', tp, gt_lesion_patches,
    #                                          tp / gt_lesion_patches, str(info['stage'].values[0]), str(info['size'].values[0]), tp, fn, fp, tn]
    #     else:
    #         gt_pancreas_patches = DataGenerator.gt_pancreas_num()
    #         patient_prediction.loc[index] = [case_id, 'healthy', fp, gt_pancreas_patches,
    #                                          fp / gt_pancreas_patches, np.nan, np.nan, tp, fn, fp, tn]

    # # Test data
    # probs = model.predict_proba(test_X)
    # patch_fig = plot_roc(probs, test_y)
    # plt.savefig(os.path.join(result_path, 'patch_roc.png'))

    # patch_fpr, patch_tpr, patch_thresholds = roc_curve(test_y, probs)
    # roc_auc = auc(patch_fpr, patch_tpr)
    # logging.info('Patch-based AUC: {}'.format(roc_auc))

    # probs = predict_binary(probs, patch_threshold)
    # patch_matrix = confusion_matrix(test_y, probs, labels=[1, 0])
    # print(patch_matrix)
    # logging.info('TP: {}'.format(patch_matrix[0][0]))
    # logging.info('FN: {}'.format(patch_matrix[0][1]))
    # logging.info('FP: {}'.format(patch_matrix[1][0]))
    # logging.info('TN: {}'.format(patch_matrix[1][1]))
    # logging.info('accuracy: {}'.format(
    #     (patch_matrix[0][0] + patch_matrix[1][1]) / len(test_y)))
    # logging.info('sensitivity: {}'.format(patch_matrix[0][0] / np.sum(test_y)))
    # logging.info('specificity: {}'.format(
    #     patch_matrix[1][1] / (test_y.shape[0] - np.sum(test_y))))

    # # Patient-based
    # patient_prediction = pd.DataFrame(
    #     columns=['case_id', 'detected_patches', 'total_patches', 'prediction'])
    # patient_y = []
    # for index, case_id in enumerate(test_list):
    #     if case_id[:2] == 'AD' or case_id[:2] == 'NP':
    #         patient_y.append(0)
    #         X, y = patch_generator(
    #             config['dataset']['dir'], case_id, patch_size,
    #             stride=5, threshold=0.0004)
    #         test_X = np.array(X)
    #         test_X = test_X.reshape(
    #             (test_X.shape[0], test_X.shape[1], test_X.shape[2], 1))
    #         y_predict = model.predict_proba(test_X)
    #         y_predict = predict_binary(y_predict, patch_threshold)
    #         patient_prediction.loc[index] = [case_id, np.sum(
    #             y_predict), len(y_predict), np.mean(y_predict)]
    #     else:
    #         patient_y.append(1)
    #         X, y = patch_generator(
    #             config['dataset']['dir'], case_id, patch_size,
    #             stride=5, threshold=0.0004)
    #         test_X = np.array(X)
    #         test_X = test_X.reshape(
    #             (test_X.shape[0], test_X.shape[1], test_X.shape[2], 1))
    #         y_predict = model.predict_proba(test_X)
    #         y_predict = predict_binary(y_predict, patch_threshold)
    #         patient_prediction.loc[index] = [case_id, np.sum(
    #             y_predict), len(y_predict), np.mean(y_predict)]

    # patient_prediction.to_csv(
    #     os.path.join(result_path, ('patient_prediction.csv')))

    # fpr, tpr, thresholds = roc_curve(
    #     patient_y, list(patient_prediction['prediction']))
    # patient_auc = auc(fpr, tpr)
    # logging.info('Patient-based AUC: {}'.format(patient_auc))

    # patient_fig = plot_roc(list(patient_prediction['prediction']), patient_y)
    # plt.savefig(os.path.join(result_path, ('patient_roc.png')))

    # patient_probs = np.array(list(patient_prediction['prediction']))
    # patient_bin = predict_binary(patient_probs, patient_threshold)
    # patient_matrix = confusion_matrix(patient_y, patient_bin, labels=[1, 0])
    # print(patient_matrix)
    # logging.info('TP: {}'.format(patient_matrix[0][0]))
    # logging.info('FN: {}'.format(patient_matrix[0][1]))
    # logging.info('FP: {}'.format(patient_matrix[1][0]))
    # logging.info('TN: {}'.format(patient_matrix[1][1]))

    print('ALL DONE!')
