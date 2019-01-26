from comet_ml import Experiment
import argparse
import sys
import os
import time
from time import gmtime, strftime
from random import shuffle
from pprint import pprint
from tqdm import trange, tqdm

import torch
from torch import nn
from torch.utils import data
from sklearn.metrics import f1_score, classification_report

from models.net_pytorch import pred_to_01
from models.net_pytorch import *
from data_loader.data_loader import getDataloader
from utils import get_config_sha1, load_config, flatten_config_for_logging


# Parse Args
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default='./configs/base.yml', type=str, help="train configuration")
parser.add_argument("-r", "--run_name", default=None, type=str, help="run name for this experiment. (Default: time)")
args = parser.parse_args()

# Load config
config = load_config(args.config)
if args.run_name is None:
    config['run_name'] = strftime("%Y%m%d_%H%M%S", gmtime())
pprint(config)

# Env settings
os.environ["CUDA_VISIBLE_DEVICES"] = config['system']['CUDA_VISIBLE_DEVICES']
if not os.path.isdir(os.path.join(config['log']['checkpoint_dir'], config['run_name'])):
    os.mkdir(os.path.join(config['log']['checkpoint_dir'], config['run_name']))
# check availability of GPU
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

# Record experiment in Comet.ml
experiment = Experiment(api_key="fdb4jkVkz4zT8vtOYIRIb0XG7",
                        project_name="pancreas-2d", workspace="adamlin120")
experiment.log_parameters(flatten_config_for_logging(config))
experiment.add_tags([config['model']['name'], config['dataset']['dir'].split('/')[-1]])
experiment.log_asset(file_path=args.config)

# Dataset
dataloaders = getDataloader(config)

# Model Init
model = eval(config['model']['name'])()
model = model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=config['optimizer']['lr'])
criterion = nn.BCELoss()

# Train & Validate & Test
global_step = 0
for epoch in trange(config['train']['epochs'], desc='EPOCH loop', leave=False):
    experiment.log_current_epoch(epoch)
    for phase in ['train', 'val', 'test']:
        if phase == 'test' and epoch != config['train']['epochs']-1:
            continue

        isTrain = phase == 'train'
        if isTrain:
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        pbar = tqdm(enumerate(dataloaders[phase]),
                    desc='{} loop'.format(phase), leave=False,
                    total=len(dataloaders[phase]))

        for i, (local_batch, local_labels) in pbar:
            local_batch = local_batch.to(device)
            local_labels = local_labels.to(device)
            optim.zero_grad()
            with torch.set_grad_enabled(isTrain):
                y_pred = model(local_batch).view(len(local_labels))
                loss = criterion(y_pred, local_labels)
                if isTrain:
                    global_step += 1
                    loss.backward()
                    optim.step()

            y_pred_class = pred_to_01(y_pred)
            accu = torch.sum(y_pred_class == local_labels).double() / len(y_pred_class)

            running_loss += loss.item() * local_batch.size(0)
            running_corrects += accu * len(y_pred_class)

            if isTrain and (global_step % config['log']['log_interval'] == 0 or i == len(dataloaders[phase])-1):
                experiment.log_metric(phase + "_loss", loss.item(), step=global_step)
                experiment.log_metric(phase + "_accuracy", accu, step=global_step)
                pbar.set_postfix({'loss':  loss.item(), 'accuracy': accu.item()})
            pbar.update(1)

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_accu = running_corrects.double() / len(dataloaders[phase].dataset)
        tqdm.write('\nEPOCH: {}  VALADATION Loss: {}  Accu: {}\n'.format(epoch, running_loss, accu))

        if not isTrain:
            experiment.log_metric(phase + "_loss", epoch_loss, step=global_step)
            experiment.log_metric(phase + "_accuracy", epoch_accu, step=global_step)
            torch.save({
                        'config': config,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim.state_dict()
                        },
                        os.path.join(config['log']['checkpoint_dir'], config['run_name'],'{}_{}_{}.pt'.format(strftime("%Y%m%d_%H%M%S", gmtime()), epoch, global_step)))
