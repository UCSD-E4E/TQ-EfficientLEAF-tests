#!/usr/bin/env python3
"""
Script to adapt the PCEN layer of an EfficientLEAF model to a new dataset.
"""
# imports
## builtin
import argparse
import time
import os
from pathlib import Path
from collections import OrderedDict

## processing
import numpy as np
from tqdm import tqdm

## torch
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

## efficentnet pytorch implementation
# !pip install efficientnet_pytorch
from efficientnet_pytorch import EfficientNet

## internal functions
from model import AudioClassifier
from model.leaf import Leaf, PCEN
from model.efficientleaf import LogTBN, EfficientLeaf
from model.mel import STFT, MelFilter, Squeeze
from engine import train
from utils import optimizer_to, scheduler_to

# Default LEAF parameters
n_filters = 40
sample_rate = 16000
window_len = 25.0
window_stride = 10.0
min_freq = 60.0
max_freq = 7800.0

def get_args_parser():
    parser = argparse.ArgumentParser('Leaf training and evaluation script', add_help=False)

    # General options (if no --ret-network or --frontend-benchmark, trains the model)
    parser.add_argument('--ret-network', action='store_true',
                        help='Returns the network and if --data-set is not "None" also returns all dataloaders')
    parser.add_argument('--frontend-benchmark', action='store_true',
                        help='Perform a benchmark on a model (can use --resume). Saves a benchmark.txt in the models folder.')
    parser.add_argument('--benchmark-runs', default=100, type=int,
                        help='Number of run to perform the benchmark --frontend-benchmark (default: 100)')

    # General network settings
    parser.add_argument('--seed', default=0, type=int,
                        help='if seed is 0, a random seed is taken')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--resume',
                        help='Path of the saved network/optimizer to resume from, ignored for new networks')
    parser.add_argument('--cudnn-benchmark', action='store_true')
    parser.add_argument('--no-cudnn-benchmark', action='store_false', dest='cudnn_benchmark')
    parser.set_defaults(cudnn_benchmark=False)

    # Training parameters
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--batch-size-eval', default=0, type=int,
                        help='Batch size used for evaluation (default: same as --batch-size')
    parser.add_argument('--epochs', default=1000, type=int,
                        help='Maximum number of epochs (default: 1000, you will want to reduce that if running with --no-scheduler)')
    parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save_every', default=5, type=int,
                        help='Model will be saved per number of epochs')
    parser.add_argument('--test-every-epoch', action='store_true',
                        help='If given, compute test error every epoch, otherwise only at the end')

    # Optimizer parameters
    parser.add_argument('--lr', default=1e-3, type=float, metavar='LR',
                        help='Learning rate for Adam (default: 1e-3)')
    parser.add_argument('--frontend-lr-factor', default=1, type=float, metavar='FACTOR',
                        help='Learning rate factor for the frontend (default: %(default)s)')
    parser.add_argument('--adam-eps', default=1e-8, type=float, metavar='EPS',
                        help='Epsilon for Adam (default: 1e-8)')
    parser.add_argument('--warmup-steps', default=0, type=int, metavar='STEPS',
                        help='Number of update steps for a linear learning rate warmup (default: %(default)s)')

    # Scheduler parameters
    parser.add_argument('--scheduler', action='store_true')
    parser.add_argument('--no-scheduler', action='store_false', dest='scheduler')
    parser.set_defaults(scheduler=True)
    parser.add_argument('--scheduler-mode', default='loss', choices=['acc', 'loss'],
                        type=str, help='Should the scheduler focus on maximizing the acc or minimizing the loss')
    parser.add_argument('--scheduler-factor', default=0.1, type=float,
                        help='Only needed if a scheduler is used. Factor that the LR will be reduced.')
    parser.add_argument('--patience', default=10, type=int, metavar='EPOCHS',
                        help='If a scheduler is used, reduce learning rate after this many epochs without improvement (default: %(default)s)')
    parser.add_argument('--min-lr', default=1e-5, type=float, help='Only needed if a scheduler is used (default: 1e-5)')

    # Dataset parameters
    parser.add_argument('--data-path', default='', type=str,
                        help='path below which to store the datasets (defaults to current directory)')
    parser.add_argument('--data-set', default='SPEECHCOMMANDS', choices=['SPEECHCOMMANDS', 'CREMAD', 'VOXFORGE', 'NSYNTH_PITCH', 'NSYNTH_INST', 'BIRDCLEF2021', 'None'],
                        type=str, help='Which dataset to use. "None" does not load any dataset; used with --ret-network to return a network without dataloaders.')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--eval-pad', default='drop', type=str, choices=('zero', 'drop', 'overlap'),
                        help='If and how to deal with incomplete evaluation chunks: "zero" for zero-padding, "drop" for omitting them (default), "overlap" for overlapping with the previous chunk')
    parser.add_argument('--eval-overlap', default=0, type=float,
                        help='Amount or fraction of overlap between consecutive evaluation chunks (default: %(default)s)')

    # Saving parameters
    parser.add_argument('--save-best-model', default='loss', choices=['acc', 'loss'],
                        type=str,
                        help='Which metric ("acc" or "loss" ) for saving the best performing model based on of validation set ("net_best_model.pth"')
    parser.add_argument('--overwrite-save', action='store_true',
                        help='Overwrites the saved run file for every --save-every as the current run with "net_checkpoint.pth"')
    parser.add_argument('--no-overwrite-save', action='store_false',
                        help='Save the best run and all the runs between "net_e(--save-every)checkpoint.pth"',
                        dest='overwrite_save')
    parser.set_defaults(overwrite_save=True)
    parser.add_argument('--save-every', default=1, type=int, help='Interval of epochs between saving the model')

    # General frontend parameters
    parser.add_argument('--frontend', default='Leaf', type=str,
                        choices=('Leaf', 'EfficientLeaf', 'Mel'),
                        help='Frontend type (Leaf, EfficientLeaf or Mel)')
    parser.add_argument('--input-size', default=sample_rate, type=int,
                        help='How long the input excerpts are in samples (default: %(default)s)')
    parser.add_argument('--output-dir', default='',
                        help='Path where the network and tensorboard logs are saved')

    # Compression parameters
    parser.add_argument('--compression', default='PCEN', choices=['PCEN', 'TBN'],
                        type=str,
                        help='Which compression method should be used PCEN (original Leaf) or TBN. (default PCEN)')
    parser.add_argument('--pcen-learn-logs', action='store_true',
                        help="If given, learns logarithms of PCEN parameters as in PCEN paper, otherwise learns parameters directly as in Leaf paper.")
    parser.add_argument('--log1p-initial-a', default=0, type=float,
                        help="Value 'a' in log(1 + 10**a * x) for the TBN compression layer (default: 0)")
    parser.add_argument('--log1p-trainable', action='store_true',
                        help="Set parameter 'a' for the TBN compression layer as trainable (default: False)")
    parser.add_argument('--log1p-per-band', action='store_true',
                        help="Learn a separate 'a' for each frequency band (default: False)")
    parser.add_argument('--tbn-median-filter', action='store_true',
                        help="Subtract the median over time in the TBN compression method")
    parser.add_argument('--tbn-median-filter-append', action='store_true',
                        help="Append the median-filtered version as a channel instead of applying in-place")

    # EfficientLeaf parameters
    parser.add_argument('--conv-win-factor', default=3, type=float,
                        help='Multiplicator (with Bandwidth) for the kernelsize of the convolution layer for EfficientLeaf')
    parser.add_argument('--stride-factor', default=1, type=float,
                        help='Multiplicator for the stride of the convolution layer for EfficientLeaf')
    parser.add_argument('--num-groups', default=4, type=int,
                        help='Number of groups for the convolution/pooling layer for EfficientLeaf')

    parser.add_argument('--model-name', default='default',
                        help='run name for subdirectory in outputs/models and outputs/runs (may contain slashes)')
    return parser

def main(args):
    device = torch.device(args.device)

    ## fix the seed for reproducibility
    if args.seed:
        seed = args.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        # random.seed(seed)

    cudnn.benchmark = args.cudnn_benchmark # setting benchmark looks for the most efficient algorithm for your hardware

    ## init dataset and dataloader
    if args.data_set == 'CREMAD':
        from datasets.crema_d import build_dataset
    if args.data_set == 'SPEECHCOMMANDS':
        from datasets.speechcommands import build_dataset
    if args.data_set == 'VOXFORGE':
        from datasets.voxforge import build_dataset
    if args.data_set == 'NSYNTH_PITCH':
        from datasets.nsynth import build_dataset_pitch as build_dataset
    if args.data_set == 'NSYNTH_INST':
        from datasets.nsynth import build_dataset_inst as build_dataset
    if args.data_set == 'BIRDCLEF2021':
        from datasets.birdclef2021 import build_dataset

    if args.data_set != 'None':
        train_loader, val_loader, test_loader, args.nb_classes = build_dataset(args=args)
        
    
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PCEN adaptation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)