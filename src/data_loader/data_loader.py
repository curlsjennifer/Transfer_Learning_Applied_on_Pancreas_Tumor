import glob
import os

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils import data
import keras
from skimage import morphology, measure
import tqdm
# import SimpleITK as sitk
import nibabel as nib
from random import shuffle

from data_loader.patch_sampler import patch_generator
from data_loader.patch_sampler import masked_2D_sampler
from data_loader.preprocessing import (
    minmax_normalization, windowing, smoothing)
from data_loader.create_boxdata import finecut_to_thickcut


def getDataloader(config):
    case_partition = convert_csv_to_dict(config['dataset']['csv'])

    training_set = Dataset_pytorch(config['dataset']['dir'],
                                   case_partition['train'],
                                   config['dataset']['input_dim'][0])
    training_generator = data.DataLoader(
        training_set, batch_size=config['train']['batch_size'],
        shuffle=True, num_workers=config['system']['num_cpu'], pin_memory=True)

    validation_set = Dataset_pytorch(config['dataset']['dir'],
                                     case_partition['validation'],
                                     config['dataset']['input_dim'][0])
    validation_generator = data.DataLoader(
        validation_set, batch_size=config['validation']['batch_size'],
        shuffle=False, num_workers=config['system']['num_cpu'], pin_memory=True)

    test_set = Dataset_pytorch(config['dataset']['dir'],
                               case_partition['test'],
                               config['dataset']['input_dim'][0])
    test_generator = data.DataLoader(
        test_set, batch_size=config['validation']['batch_size'],
        shuffle=False, num_workers=config['system']['num_cpu'], pin_memory=True)

    dataloaders = {'train': training_generator,
                   'val': validation_generator, 'test': test_generator}

    return dataloaders


class Dataset_pytorch(data.Dataset):
    """Pytorch Dataset for 2d patch image inheriting torch.utils.data.Dataset

    Args:
        list_IDs (list): List of patch IDs
        labels (dict): {patch_id : label}
        patch_paths (dict): {patch_id : abs. path of patch}

    Attributes:
        list_IDs (list): List of patch IDs
        labels (dict): {patch_id : label}
        patch_paths (dict): {patch_id : abs. path of patch}
        load_fn(funciton): loading function for images

    """

    def __init__(self, data_path, case_list, patch_size, return_id=False):
        self.case_list = case_list
        self.data_path = data_path
        self.patch_size = patch_size
        self.return_id = return_id

        self.X, self.labels = load_patches(self.data_path,
                                           self.case_list,
                                           self.patch_size)
        # convert to channel first
        self.X = np.transpose(self.X, (0, 3, 1, 2))

    def __len__(self):
        """Get number of data in this dataset

        Returns:
            int: number of data in this dataset
        """
        return len(self.labels)

    def __getitem__(self, index):
        """Get a data in dataset

        The image is read from disk at patch_paths given the index.

        Args:
            index (int): index of data

        Returns:
            X (np array): image of shape (1, image_height, image_width)
            y (int): lable

        """
        # Load data and get label
        X = torch.from_numpy(self.X[index]).to(torch.float)  # channel first
        y = torch.tensor(self.labels[index]).to(torch.float)

        if self.return_id:
            return X, y, ID
        return X, y


class DataGenerator_keras(keras.utils.Sequence):
    """Keras Dataset for 2d patch image inheriting keras.utils.Sequence

    Need multi-thread to avoid bottlebecking at Disk IO since images are read from disk.

    Args:
        list_IDs (list): List of patch IDs
        labels (dict): {patch_id : label}
        patch_paths (dict): {patch_id : abs. path of patch}

        batch_size (int): number of data in each step
        dim (tuple): dimension of iamge
        n_channels=1 (int): number of image channel
        n_classes (int): number of classes
        shuffle (bool): if shuffle data on epoch ends
        load_fn (func): loading function for images

    Attributes:
        same as args

    """

    def __init__(self, X_total, y_total, list_IDs, labels, patch_paths,
                 batch_size=32, dim=(32, 32), n_channels=1,
                 n_classes=2, shuffle=False, load_fn=np.load):
        'Initialization'
        self.X_total = X_total
        self.y_total = y_total

        self.list_IDs = list_IDs
        self.labels = labels
        self.patch_paths = patch_paths

        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle

        self.load_fn = load_fn

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index
                               * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(indexes)
        # X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels),
                     dtype=np.float32)  # channel last
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, index in enumerate(indexes):
            # Store sample
            X[i, ] = self.X_total[index]  # channel last

            # Store class
            y[i] = self.y_total[index]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    # def __data_generation(self, list_IDs_temp):
    #     # X : (n_samples, *dim, n_channels)
    #     'Generates data containing batch_size samples'
    #     # Initialization
    #     X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)  # channel last
    #     y = np.empty((self.batch_size), dtype=int)

    #     # Generate data
    #     for i, ID in enumerate(list_IDs_temp):
    #         # Store sample
    #         X[i, ] = self.load_fn(self.patch_paths[ID])[:, :, np.newaxis]  # channel last

    #         # Store class
    #         y[i] = self.labels[ID]

    #     return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


class DataGenerator_other():
    def __init__(self, data_path, patch_size, stride=5, data_type='MSD'):
        self.data_path = data_path
        self.patch_size = patch_size
        self.stride = stride
        self.data_type = data_type

    def load_image(self, filename):
        if self.data_type == 'Pancreas-CT':
            file_location = glob.glob(os.path.join(
                self.data_path, filename) + '/*/*/000001.dcm')
            imagedir = os.path.dirname(file_location[0])
            labelname = 'label' + filename[-4:] + '.nii.gz'
            labelpath = os.path.join(self.data_path, 'annotations', labelname)

            # reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(imagedir)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            # image_array = sitk.GetArrayFromImage(image).transpose((2, 1, 0))
            self.thickness = image.GetSpacing()[2]
            label = nib.load(labelpath).get_data()
        elif self.data_type == 'MSD':
            imagepath = os.path.join(self.data_path, 'imagesTr', filename)
            labelpath = os.path.join(self.data_path, 'labelsTr', filename)

            image_array = nib.load(imagepath).get_data()
            label = nib.load(labelpath).get_data()
            self.affine = nib.load(imagepath).affine
            self.thickness = self.affine[2, 2]

        return image_array, label

    def get_boxdata(self, filename, border=(10, 10, 3)):

        image, label = self.load_image(filename)

        pancreas = np.zeros(label.shape)
        pancreas[np.where(label != 0)] = 1
        lesion = np.zeros(label.shape)
        if self.data_type == 'MSD':
            lesion[np.where(label == 2)] = 1

        if self.thickness < 5:
            image = finecut_to_thickcut(image, self.thickness)
            pancreas = finecut_to_thickcut(
                pancreas, self.thickness, label_mode=True)
            lesion = finecut_to_thickcut(
                lesion, self.thickness, label_mode=True)

        xmin, ymin, zmin = np.max(
            [np.min(np.where(pancreas != 0), 1) - border, (0, 0, 0)], 0)
        xmax, ymax, zmax = np.min(
            [np.max(np.where(pancreas != 0), 1) + border, label.shape], 0)

        box_image = image[xmin:xmax, ymin:ymax, zmin:zmax]
        box_pancreas = pancreas[xmin:xmax, ymin:ymax, zmin:zmax]
        box_lesion = lesion[xmin:xmax, ymin:ymax, zmin:zmax]

        return box_image, box_pancreas, box_lesion

    def preprocessing(self, filename):
        from skimage import morphology
        image, pancreas, lesion = self.get_boxdata(filename)

        if self.data_type == 'Pancreas-CT':
            image = image[:, ::-1, :]
            pancreas = pancreas[:, ::-1, :]
            lesion = lesion[:, ::-1, :]
            pancreas = smoothing(pancreas)
            lesion = smoothing(lesion)
        elif self.data_type == 'MSD':
            image = image[::-1, ::-1, :]
            pancreas = pancreas[::-1, ::-1, :]
            lesion = lesion[::-1, ::-1, :]

        pancreas = morphology.dilation(pancreas, np.ones([3, 3, 1]))
        lesion = morphology.dilation(lesion, np.ones([3, 3, 1]))

        image = windowing(image)
        image = minmax_normalization(image)

        return image, pancreas, lesion

    def generate_patch(self, filename):
        X = []
        Y = []

        self.box_image, self.box_pancreas, self.box_lesion = self.preprocessing(
            filename)

        self.coords = masked_2D_sampler(self.box_lesion, self.box_pancreas,
                                        self.patch_size, self.stride, threshold=1 / (self.patch_size ** 2))

        self.box_image[np.where(self.box_pancreas == 0)] = 0

        for coord in self.coords:
            mask_pancreas = self.box_image[coord[1]
                                           :coord[4], coord[2]:coord[5], coord[3]]
            X.append(mask_pancreas)
            Y.append(coord[0])

        self.X = X
        self.Y = Y

        return X, Y

    def patch_len(self):
        return len(self.X)

    def gt_pancreas_num(self):
        return len(self.Y) - np.sum(self.Y)

    def gt_lesion_num(self):
        return np.sum(self.Y)

    def get_prediction(self, filename, model, patch_threshold=0.5):
        from sklearn.metrics import confusion_matrix
        X, Y = self.generate_patch(filename)
        if len(self.X) > 0:
            test_X = np.array(self.X)
            test_X = test_X.reshape(
                (test_X.shape[0], test_X.shape[1], test_X.shape[2], 1))
            test_Y = np.array(self.Y)
        else:
            return filename

        self.probs = model.predict_proba(test_X)
        predict_y = predict_binary(self.probs, patch_threshold)
        self.patch_matrix = confusion_matrix(test_Y, predict_y, labels=[1, 0])

        return self.patch_matrix

    def get_auc(self):
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(self.Y, self.probs)
        return auc(fpr, tpr)

    def get_probs(self):
        return self.probs, self.Y

    def get_roc_curve(self):
        from data_description.visualization import plot_roc
        plot_roc(self.probs, self.Y)

    def get_all_value(self):
        tp = self.patch_matrix[0][0]
        fn = self.patch_matrix[0][1]
        fp = self.patch_matrix[1][0]
        tn = self.patch_matrix[1][1]
        return tp, fn, fp, tn


class DataGenerator_NTUH():
    def __init__(self, data_path, patch_size, stride=5):
        self.data_path = data_path
        self.patch_size = patch_size
        self.stride = stride

    def load_image(self, case_id, backup_path=''):
        imagepath = os.path.join(
            self.data_path, 'image', 'IM_' + case_id + '.nii.gz')
        labelpath = os.path.join(
            self.data_path, 'label', 'LB_' + case_id + '.nii.gz')
        rothpath = os.path.join(backup_path, 'IM_' + case_id + '/IM_' + case_id + '_model.nii.gz')

        image_array = nib.load(imagepath).get_data()
        if os.path.exists(labelpath):
            label = nib.load(labelpath).get_data()
        elif os.path.exists(rothpath):
            result = nib.load(rothpath).get_data()
            label = np.zeros(result.shape)
            label[np.where(result == 8)] = 1
        else:
            print("can't find label file!")
        self.affine = nib.load(imagepath).affine
        self.thickness = self.affine[2, 2]

        return image_array, label

    def load_boxdata(self, case_id):
        imagepath = os.path.join(self.data_path, case_id, 'ctscan.npy')
        pancreaspath = os.path.join(self.data_path, case_id, 'pancreas.npy')
        lesionpath = os.path.join(self.data_path, case_id, 'lesion.npy')
        image = np.load(imagepath)
        pancreas = np.load(pancreaspath)
        lesion = np.load(lesionpath)

        return image, pancreas, lesion

    def get_boxdata(self, image, label, border=(20, 20, 3)):

        pancreas = np.zeros(label.shape)
        pancreas[np.where(label != 0)] = 1
        lesion = np.zeros(label.shape)
        lesion[np.where(label == 2)] = 1

        if self.thickness < 5:
            image = finecut_to_thickcut(image, self.thickness)
            pancreas = finecut_to_thickcut(
                pancreas, self.thickness, label_mode=True)
            lesion = finecut_to_thickcut(
                lesion, self.thickness, label_mode=True)

        xmin, ymin, zmin = np.max(
            [np.min(np.where(pancreas != 0), 1) - border, (0, 0, 0)], 0)
        xmax, ymax, zmax = np.min(
            [np.max(np.where(pancreas != 0), 1) + border, label.shape], 0)

        box_image = image[xmin:xmax, ymin:ymax, zmin:zmax]
        box_pancreas = pancreas[xmin:xmax, ymin:ymax, zmin:zmax]
        box_lesion = lesion[xmin:xmax, ymin:ymax, zmin:zmax]

        return box_image, box_pancreas, box_lesion

    def preprocessing(self, image, pancreas, lesion):
        from skimage import morphology

        image = image[::-1, ::-1, :]
        pancreas = pancreas[::-1, ::-1, :]
        lesion = lesion[::-1, ::-1, :]

        pancreas = morphology.dilation(pancreas, np.ones([3, 3, 1]))
        lesion = morphology.dilation(lesion, np.ones([3, 3, 1]))

        image = windowing(image)
        image = minmax_normalization(image)

        return image, pancreas, lesion

    def generate_patch(self, image, pancreas, lesion):
        X = []
        Y = []

        self.coords = masked_2D_sampler(lesion, pancreas,
                                        self.patch_size, self.stride, threshold=1 / (self.patch_size ** 2))

        image[np.where(pancreas == 0)] = 0

        for coord in self.coords:
            mask_pancreas = image[coord[1]
                                  :coord[4], coord[2]:coord[5], coord[3]]
            X.append(mask_pancreas)
            Y.append(coord[0])

        self.X = X
        self.Y = Y

        return X, Y

    def patch_len(self):
        return len(self.X)

    def gt_pancreas_num(self):
        return len(self.Y) - np.sum(self.Y)

    def gt_lesion_num(self):
        return np.sum(self.Y)

    def get_prediction(self, model, patch_threshold=0.5):
        from sklearn.metrics import confusion_matrix
        test_X = np.array(self.X)
        test_X = test_X.reshape(
            (test_X.shape[0], test_X.shape[1], test_X.shape[2], 1))
        test_Y = np.array(self.Y)

        self.probs = model.predict_proba(test_X)
        predict_y = predict_binary(self.probs, patch_threshold)
        self.patch_matrix = confusion_matrix(test_Y, predict_y, labels=[1, 0])

        return self.patch_matrix

    def get_auc(self):
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(self.Y, self.probs)
        return auc(fpr, tpr)

    def get_roc_curve(self):
        from data_description.visualization import plot_roc
        plot_roc(self.probs, self.Y)

    def get_probs(self):
        return self.probs, self.Y

    def get_all_value(self):
        tp = self.patch_matrix[0][0]
        fn = self.patch_matrix[0][1]
        fp = self.patch_matrix[1][0]
        tn = self.patch_matrix[1][1]
        return tp, fn, fp, tn


def predict_binary(prob, threshold):
    binary = np.zeros(prob.shape)
    binary[prob < threshold] = 0
    binary[prob >= threshold] = 1
    return binary


def split_save_case_partition(case_list, ratio=(0.8, 0.1, 0.1), path=None, test_cases=None, random_seed=None):
    """Splting all cases to train, val, test part

    If path is not empty str, partition dict is saved for reproducibility.

    Args:
        case_list (list): The list contains case name.
        ratio (tup): Data split ratio. SHOULD sum to 1. (train, val, test) Defaults to (.8, .1, .1).
        path (str): Path to Save the partition dict for reproducibility.
        test_cases (list): For fixing the testing cases.
        random_seed (int): Random Seed.


    Returns:
        dict:   Keys: {all, train, validation, test}
                Values: list of cases

    """

    print('SPLIT_SAVE_CASE_PARTITION:\tStart spliting cases...')

    partition = {}
    partition['all'] = case_list
    if test_cases is None:
        # load case list and spilt to 3 part
        print('SPLIT_SAVE_CASE_PARTITION:\tTarget Partition Ratio: (train, val, test)={}'.format(ratio))
        partition['train'], partition['test'] = train_test_split(
            partition['all'], test_size=ratio[2], random_state=random_seed)
        partition['train'], partition['validation'] = train_test_split(
            partition['train'], test_size=ratio[1] / (ratio[0] + ratio[1]), random_state=random_seed)
    elif type(test_cases) is list:
        # load predifined test cases
        print('SPLIT_SAVE_CASE_PARTITION:\tUsing PREDEFINED TEST CASES')
        partition['validation'] = test_cases
        partition['test'] = []
        partition['train'] = list(set(case_list) - set(test_cases))
    else:
        raise TypeError(
            "test_cases expected to be \"list\", instead got {}".format(type(test_cases)))

    # report actual partition ratio
    num_parts = list(map(len, [partition[part]
                               for part in ['train', 'validation', 'test']]))
    real_ratio = np.array(num_parts) / sum(num_parts)
    print('SPLIT_SAVE_CASE_PARTITION:\tActual Partition Ratio: (train, val, test)={}'.format(
        (real_ratio)))

    print('SPLIT_SAVE_CASE_PARTITION:\tDone Partition')
    # saving partition dict to disk
    if path is not None:
        print('SPLIT_SAVE_CASE_PARTITION:\tStart saving partition dict to {}'.format(path))
        with open(path, 'wb') as f:
            pickle.dump(partition, f, pickle.HIGHEST_PROTOCOL)
        print('SPLIT_SAVE_CASE_PARTITION:\tDone saving')

    return partition


def fix_save_case_partition(case_list, ratio=(0.8, 0.1, 0.1), path='', random_seed=None):
    """Fixing test case

    TODO: ratio
    Splting all cases to train, val, test part

    If path is not empty str, partition dict is saved for reproducibility.

    Args:
        case_list (list): The list contains case name.
        ratio (tup): Data split ratio. SHOULD sum to 1. (train, val, test) Defaults to (.8, .1, .1).
        path (str): Path to Save the partition dict for reproducibility.
        random_seed (int): Random Seed.


    Returns:
        dict:   Keys: {all, train, validation, test}
                Values: list of cases

    """

    print('SPLIT_SAVE_CASE_PARTITION:\tStart spliting cases...')

    # load case list and spilt to 3 part
    partition = {}
    partition['all'] = case_list
    partition['test'] = ['NP4', 'NP8', 'NP2', 'NP9', 'NP10',
                         'AD20', 'AD110', 'AD29', 'AD92', 'AD87',
                         'PT13', 'PT35', 'PT2', 'PT36', 'PT42',
                         'PC83', 'PC39', 'PC79', 'PC73', 'PC72']
    partition['train'] = [i for i in partition['all']
                          if i not in partition['test']]
    partition['train'], partition['validation'] = train_test_split(
        partition['train'], test_size=ratio[1] / (ratio[0] + ratio[1]),
        random_state=random_seed)

    # report actual partition ratio
    num_parts = list(map(len, [partition[part]
                               for part in ['train', 'validation', 'test']]))
    real_ratio = np.array(num_parts) / sum(num_parts)

    print('SPLIT_SAVE_CASE_PARTITION:\tDone Partition')
    # saving partition dict to disk
    if path != "":
        print('SPLIT_SAVE_CASE_PARTITION:\tStart saving partition dict to {}'.format(path))
        with open(path, 'wb') as f:
            pickle.dump(partition, f, pickle.HIGHEST_PROTOCOL)
        print('SPLIT_SAVE_CASE_PARTITION:\tDone saving')

    return partition


def load_case_partition(path):
    """Load the cases partition

    Args:
        path (str): Path to partition dict.


    Returns:
        dict:   Keys: {all, train, validation, test}
                Values: list of cases

    """

    print('LOAD_CASE_PARTITION:\tStart loading case partition...')

    # loading partition dict from disk
    with open(path, 'rb') as f:
        partition = pickle.load(f)

    # report partiton ratio
    num_parts = list(map(len, [partition[part]
                               for part in ['train', 'validation', 'test']]))
    ratio = num_parts / sum(num_parts)
    print('LOAD_CASE_PARTITION:\tPartition Ratio: (train, val, test)={}'.format((ratio)))

    print('LOAD_CASE_PARTITION:\tDone loading case partition')
    return partition


def get_patch_partition_labels(case_partition, pancreas_dir, lesion_dir):
    """Splting patches based on case partition.

    patch_id = case_pathIndex


    Returns:
        dict:   Keys: {all, train, validation, test}
                Values: list of patch id
        dict:   Keys: patch_id
                Values: abs. path of patch
        dict:   Keys: patch_id
                Values: label

    """

    patch_partition = {'train': [], 'validation': [], 'test': []}
    patch_paths = {'train': [], 'validation': [], 'test': []}
    labels = {}

    print('GET_PATCH_PARTITION_LABELS:\tStart loading patch partition...')
    for part in ['train', 'validation', 'test']:
        print('GET_PATCH_PARTITION_LABELS:\t Progress: {}'.format(part))
        for case in case_partition[part]:
            for i, path_dir in enumerate([pancreas_dir, lesion_dir]):
                for patch_path in glob.glob(path_dir + '/' + case + '_*.npy'):
                    patch_id = patch_path.split('/')[-1].split('.')[0]
                    patch_partition[part].append(patch_id)
                    patch_paths[patch_id] = patch_path
                    labels[patch_id] = i
    print('GET_PATCH_PARTITION_LABELS:\tDone patch partition')
    return patch_partition, patch_paths, labels


def load_patches(data_path, case_list, patch_size=50, stride=5):
    X_total = []
    y_total = []
    for ID in tqdm.tqdm(case_list):
        X_tmp, y_tmp = patch_generator(
            data_path, ID, patch_size, stride=stride, threshold=0.0004, max_amount=1000)
        X_total.extend(X_tmp)
        y_total.extend(y_tmp)
    X = np.array(X_total)
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
    y = np.array(y_total)

    return X, y


def convert_csv_to_dict(csv_data_path='/data2/pancreas/raw_data/data_list.csv'):
    """
    version: 2019/03
    extract data list of train, validation, test from csv
    csv_data_path = path to the csv file (ex: '/home/d/pancreas/raw_data/data_list.csv')
    Returns:
        dict:   Keys: {all, train, validation, test}
                Values: list of patch id
    """
    final_split_df = pd.read_csv(csv_data_path)
    data_list_dict = {}
    data_list_dict['train'] = list(
        final_split_df[final_split_df['Class'] == 'train']['Number'])
    data_list_dict['validation'] = list(
        final_split_df[final_split_df['Class'] == 'validation']['Number'])
    data_list_dict['test'] = list(
        final_split_df[final_split_df['Class'] == 'test']['Number'])
    data_list_dict['all'] = list(final_split_df[final_split_df['Class'] == 'train']['Number']) + \
        list(final_split_df[final_split_df['Class'] == 'validation']['Number']) + \
        list(final_split_df[final_split_df['Class'] == 'test']['Number'])
    print('Finish converting csv to dict')
    return data_list_dict


def load_list(list_path):
    '''
    version: 2019/08
    extract data list of train and test from csv
    training data: 100 healthy and 100 tumor
    testing data: 80 healthy and 80 tumor
    '''
    df = pd.read_csv(list_path, converters={'add_date': str})
    data_list_dict = {}

    healthy_total = df[(df['type'] == 'healthy') & (df['diff_patient_list'] == True)]
    healthy_train = list(
        healthy_total[healthy_total['add_date'] == '20190210']['case_id'])
    healthy_train.remove('AD54')
    healthy_train.remove('AD95')
    healthy_test = list(
        healthy_total[healthy_total['add_date'] == '20190618']['case_id'])
    healthy_test.remove('AD137')
    healthy_test.remove('AD143')

    tumor_total = df[(df['type'] == 'tumor') & (df['diff_patient_list'] == True)]
    tumor_train = list(
        tumor_total[tumor_total['add_date'] == '20190210']['case_id'])
    tumor_train.remove('PC47')
    tumor_test = list(tumor_total[tumor_total['add_date'] == '20190618']['case_id'])
    tumor_test.remove('PC570')
    tumor_test.remove('PC653')

    shuffle(healthy_train)
    shuffle(tumor_train)

    data_list_dict['healthy_train'] = healthy_train
    data_list_dict['healthy_test'] = healthy_test
    data_list_dict['tumor_train'] = tumor_train
    data_list_dict['tumor_test'] = tumor_test

    return data_list_dict
