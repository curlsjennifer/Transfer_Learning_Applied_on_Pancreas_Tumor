import sys
import os
import argparse
import multiprocessing
from pprint import pprint
from ast import literal_eval
from tqdm import trange, tqdm

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from models.net_pytorch import pred_to_01
from models.net_pytorch import *
from data_loader.data_loader import split_save_case_partition, load_case_partition, get_patch_partition_labels, Dataset_pytorch


def calculate_metrics(case_base, case, case_df, test_cases):
        TP = len(case_df[(case_df.Y==1) & (case_df.pred_class==1)])
        FP = len(case_df[(case_df.Y==0) & (case_df.pred_class==1)])
        FN = len(case_df[(case_df.Y==1) & (case_df.pred_class==0)])
        TN = len(case_df[(case_df.Y==0) & (case_df.pred_class==0)])
        con_mat=[[TN,FP],[FN,TP]]

        TPR = float(TP) / (TP + FN) if TP + FN != 0 else -1
        TNR = float(TN) / (TN + FP) if TN + FP != 0 else -1
        PPV = float(TP) / (TP + FP) if TP + FP != 0 else -1
        F1 = 2.0 * TP / (2.0 * TP + FN + FP) if 2.0 * TP + FN + FP != 0 else -1
        accuracy = float(TP + TN) / (TP + FP + FN + TN)


        case_base['case'].append(case)
        case_base['type'].append('lesion' if any(case_df.Y==1) else 'pancreas')
        case_base['num_lesion_patch'].append(int(sum(case_df.Y)))
        case_base['isTrain'].append('test' if case in test_cases else 'train')
        case_base['accuracy'].append(accuracy)
        case_base['precision'].append(PPV)
        case_base['sensitivity'].append(TPR)
        case_base['specificity'].append(TNR)
        case_base['f1'].append(F1)
        case_base['confusion_matrix'].append(con_mat)


def case_base_dataframe(df, test_cases):
    case_base = {'case': [], 'type': [], 'num_lesion_patch': [], 'isTrain': [], 'accuracy': [], 'precision': [], 'sensitivity': [], 'specificity': [], 'f1': [], 'confusion_matrix': []}

    # all case
    calculate_metrics(case_base, 'ALL', df, test_cases)
    # single case
    for case in set([x[0] for x in df.index.values]):
        calculate_metrics(case_base, case, df.loc[case], test_cases)

    case_base_df = (pd.DataFrame(case_base, 
                                index=case_base['case'])
                    .sort_values('num_lesion_patch', ascending=False)
                    .drop(columns=['case'])
                   )
    return case_base_df


# argument parser
parser = argparse.ArgumentParser(description='Testing of pancrease 2d classification model')
parser.add_argument('-c', '--ckpt_path', help='model checkpoint path')
parser.add_argument('--test_cases', help='test cases list', nargs='+', default=[])
parser.add_argument('--case_base_report', help='file for save case base report', default=None)

parser.add_argument('--gpu_device', '-g', help='cuda device index', default="")
parser.add_argument('--batch_size', help='batch size', type=int, default=128)

parser.add_argument('--case_list_dir', help='case list dir', default="/home/d/pancreas/box_data_extract")
parser.add_argument('--patch_pancreas_dir', help='cuda device index', default="/home/d/pancreas/patch_data/withmask/pancreas")
parser.add_argument('--patch_lesion_dir', help='cuda device index', default="/home/d/pancreas/patch_data/withmask/lesion")

args = parser.parse_args()
print(args)

# Loading checkpoint: {config, epoch, model_state_dict, optim_state_dict}
checkpoint = torch.load(args.ckpt_path)
config = checkpoint['config']
args.case_list_dir = config['case_list_dir']
args.patch_pancreas_dir = config['patch_pancreas_dir']
args.patch_lesion_dir = config['patch_lesion_dir']

# Env settings
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

case_partition = {'train': [], 'validation': [], 'test': os.listdir(args.case_list_dir)}
patch_partition, patch_paths, labels = get_patch_partition_labels(
    case_partition, args.patch_pancreas_dir, args.patch_lesion_dir)

# Data Generators
test_set = Dataset_pytorch(patch_partition['test'], labels, patch_paths, return_id=True)
test_generator = data.DataLoader(
    test_set, batch_size=args.batch_size, shuffle=True, num_workers=multiprocessing.cpu_count(), pin_memory=True)

# Model loadingx
model = eval(config['model'])()
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

# Metrics
case_pred = {case_name: [] for case_name in args.case_list_dir}

# Loop over epochs
model.eval()
pred, pred_class, Y = torch.tensor([]), torch.tensor([]), torch.tensor([])
IDs = []
for local_batch, local_labels, id in tqdm(test_generator, desc='TEST loop', leave=False):
    # Transfer to GPU
    local_batch = local_batch.to(device)

    # Model computations
    local_pred = model(local_batch).view(len(local_labels))

    pred = torch.cat((pred, local_pred.clone().detach().cpu()))
    pred_class = torch.cat((pred_class, pred_to_01(local_pred).clone().detach().cpu()))
    Y = torch.cat((Y, local_labels))
    IDs += id


accu = accuracy_score(Y, pred_class)

print('Accu: {}\n'.format(accu))
print(classification_report(Y, pred_class, target_names=['pancreas', 'leison']))
print(confusion_matrix(Y, pred_class))

cases = [id.split('_')[0] for id in IDs]

df = pd.DataFrame(
        {"Y": Y,
         "pred_class": pred_class,
         "pred": pred},
        index=pd.MultiIndex.from_tuples(zip(cases, IDs), names=['case', 'patch_id'])).sort_index()
df_case = case_base_dataframe(df, args.test_cases)

print(df)
print(df_case)

if args.case_base_report is not None:
    df_case.to_pickle(args.case_base_report)
