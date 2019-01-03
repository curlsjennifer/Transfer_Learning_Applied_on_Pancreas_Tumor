from comet_ml import Experiment
import sys
import os
from random import shuffle
from pprint import pprint
from ast import literal_eval
from tqdm import trange, tqdm

import torch
from torch import nn
from torch.utils import data
from sklearn.metrics import f1_score, classification_report

from models.net_pytorch import pred_to_01
from models.net_pytorch import *
from data_loader.data_loader import split_save_case_partition, load_case_partition, get_patch_partition_labels, Dataset_pytorch
from utils import get_config_sha1


# Load config
with open(sys.argv[1], 'r') as f:
    config = literal_eval(f.read())
    config['config_sha1'] = get_config_sha1(config, 5)
    pprint(config)

# Env settings
os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_VISIBLE_DEVICES']
# check availability of GPU
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

# Record experiment in Comet.ml
experiment = Experiment(api_key="fdb4jkVkz4zT8vtOYIRIb0XG7",
                        project_name="pancreas-2d", workspace="adamlin120")
experiment.log_parameters(config)
experiment.add_tag(config['model'])
experiment.add_tag(config['patch_pancreas_dir'].split('/')[-2])

# split cases into train, val, test
case_list = os.listdir(config['case_list_dir'])
case_partition = split_save_case_partition(case_list, config['case_split_ratio'], path=config['case_partition_path'],
                                           test_cases=config['test_list'], random_seed=config['random_seed'])

# Get patch partition
patch_partition, patch_paths, labels = get_patch_partition_labels(
    case_partition, config['patch_pancreas_dir'], config['patch_lesion_dir'])

# Data Generators
training_set = Dataset_pytorch(patch_partition['train'], labels, patch_paths)
training_generator = data.DataLoader(
    training_set, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_cpu'], pin_memory=True)

validation_set = Dataset_pytorch(patch_partition['validation'], labels, patch_paths)
validation_generator = data.DataLoader(
    validation_set, batch_size=config['val_batch_size'], shuffle=False, num_workers=config['num_cpu'], pin_memory=True)

# Model Init
model = eval(config['model'])()
model = model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=config['lr'])
criterion = nn.BCELoss()

# Loop over epochs
global_step = 0
for epoch in trange(config['epochs'], desc='EPOCH loop', leave=False):
    # Training
    with experiment.train():
        model.train()
        experiment.log_current_epoch(epoch)
        num_correct, num_count, running_loss, accu = 0.0, 0.0, 0.0, 0.0
        pbar = tqdm(enumerate(training_generator),
                    desc='TRAIN loop', leave=False)
        for i, (local_batch, local_labels) in pbar:
            global_step += 1
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(
                device), local_labels.to(device)

            # Model computations
            optim.zero_grad()
            y_pred = model(local_batch).view(len(local_labels))
            loss = criterion(y_pred, local_labels)
            loss.backward()
            optim.step()

            running_loss += loss.item() / len(local_labels)
            y_pred_class = pred_to_01(y_pred)
            num_correct += (y_pred_class.clone().detach().cpu()
                            == local_labels.clone().detach().cpu()).sum()
            num_count += len(local_labels)
            if global_step % 1 == 0:
                accu = float(num_correct) / num_count
                f1 = f1_score(local_labels.clone().detach().cpu(
                ), y_pred_class.clone().detach().cpu(), average='macro')
                pbar.set_description('Loss: {0:.3f}'.format(running_loss))
                pbar.refresh()
                experiment.log_metric("loss", running_loss, step=global_step)
                experiment.log_metric("accuracy", accu, step=global_step)
                experiment.log_metric("f1", f1, step=global_step)
                num_correct, num_count, running_loss = 0.0, 0.0, 0.0

    # Validation
    with experiment.test():
        with torch.set_grad_enabled(False):
            model.eval()
            num_correct, num_count, running_loss = 0., 0., 0.
            pred = torch.tensor([])
            Y = torch.tensor([])
            for local_batch, local_labels in tqdm(validation_generator, desc='VAL loop', leave=False):
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(
                    device), local_labels.to(device)

                # Model computations
                y_pred = model(local_batch).view(len(local_labels))
                loss = criterion(y_pred, local_labels)

                running_loss += loss.item() / len(local_labels)
                y_pred_class = pred_to_01(y_pred)
                num_correct += (y_pred_class.clone().detach().cpu()
                                == local_labels.clone().detach().cpu()).sum()
                num_count += len(local_labels)

                pred = torch.cat((pred, y_pred_class.clone().detach().cpu()))
                Y = torch.cat((Y, local_labels.clone().detach().cpu()))
            accu = float(num_correct) / num_count
            f1 = f1_score(Y.clone().detach().cpu(),
                          pred.clone().detach().cpu(), average='macro')
            tqdm.write(classification_report(Y, pred))
            tqdm.write('\nEPOCH: {}  VALADATION Loss: {}  Accu: {} F1: {}\n'.format(
                epoch, running_loss, accu, f1))
            experiment.log_metric("loss", running_loss, step=global_step)
            experiment.log_metric("accuracy", accu, step=global_step)
            experiment.log_metric("f1", f1, step=global_step)
