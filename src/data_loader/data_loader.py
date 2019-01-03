import glob
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils import data
import keras


class Dataset_pytorch(data.Dataset):
    """Pytorch Dataset for 2d patch image inheriting torch.utils.data.Dataset

    Need multi-thread to avoid bottlebecking at Disk IO since images are read from disk.

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

    def __init__(self, list_IDs, labels, patch_paths, load_fn=np.load):
        self.labels = labels
        self.list_IDs = list_IDs
        self.patch_paths = patch_paths
        self.load_fn = load_fn

    def __len__(self):
        """Get number of data in this dataset

        Returns:
            int: number of data in this dataset
        """
        return len(self.list_IDs)

    def __getitem__(self, index):
        """Get a data in dataset

        The image is read from disk at patch_paths given the index.

        Args:
            index (int): index of data

        Returns:
            X (np array): image of shape (1, image_height, image_width)
            y (int): lable

        """
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.from_numpy(self.load_fn(self.patch_paths[ID])[
                             np.newaxis, :, :]).to(torch.float)  # channel first
        y = torch.tensor(self.labels[ID]).to(torch.float)

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

    def __init__(self, list_IDs, labels, patch_paths, batch_size=32, dim=(32, 32), n_channels=1,
                 n_classes=2, shuffle=False, load_fn=np.load):
        'Initialization'
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
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)  # channel last
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i, ] = self.load_fn(self.patch_paths[ID])[:, :, np.newaxis]  # channel last

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


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
        raise TypeError("test_cases expected to be \"list\", instead got {}".format(type(test_cases)))

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
            for i, path_dir in enumerate([lesion_dir, pancreas_dir]):
                for patch_path in glob.glob(path_dir + '/' + case + '_*.npy'):
                    patch_id = patch_path.split('/')[-1].split('.')[0]
                    patch_partition[part].append(patch_id)
                    patch_paths[patch_id] = patch_path
                    labels[patch_id] = i
    print('GET_PATCH_PARTITION_LABELS:\tDone patch partition')
    return patch_partition, patch_paths, labels
