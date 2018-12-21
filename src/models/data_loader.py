import glob
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, list_IDs, labels, patch_paths):
        self.labels = labels
        self.list_IDs = list_IDs
        self.patch_paths = patch_paths

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.from_numpy(np.load(self.patch_paths[ID])[np.newaxis, :, :]).to(torch.float)
        y = torch.tensor(self.labels[ID]).to(torch.float)

        return X, y


def split_save_case_partition(case_list, ratio=(0.8, 0.1, 0.1), path='', random_seed=None):
    """Splting all cases to train, val, test part

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
    print('SPLIT_SAVE_CASE_PARTITION:\tTarget Partition Ratio: (train, val, test)={}'.format(ratio))
    partition = {}
    partition['all'] = case_list
    partition['train'], partition['test'] = train_test_split(
        partition['all'], test_size=ratio[2], random_state=random_seed)
    partition['train'], partition['validation'] = train_test_split(
        partition['train'], test_size=ratio[1]/(ratio[0]+ratio[1]), random_state=random_seed)

    # report actual partition ratio
    num_parts = list(map(len, [partition[part]
                               for part in ['train', 'validation', 'test']]))
    real_ratio = np.array(num_parts) / sum(num_parts)
    print('SPLIT_SAVE_CASE_PARTITION:\tActual Partition Ratio: (train, val, test)={}'.format(
        (real_ratio)))

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
            for i, path_dir in enumerate([lesion_dir, pancreas_dir]):
                for patch_path in glob.glob(path_dir+'/'+case+'_*.npy'):
                    patch_id = patch_path.split('/')[-1].split('.')[0]
                    patch_partition[part].append(patch_id)
                    patch_paths[patch_id] = patch_path
                    labels[patch_id] = i
    print('GET_PATCH_PARTITION_LABELS:\tDone patch partition')
    return patch_partition, patch_paths, labels
