import os
import random 
import argparse

import numpy as np

import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='4,5')

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr_decay', type=str, default='step')
    parser.add_argument('--warmup_epochs', type=int, default=10)    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--decay_step', type=int, default=50)
    
    parser.add_argument('--use_fp16', type=bool, default=True)
    parser.add_argument('--use_grad_accumulate', type=bool, default=True)
    
    # Settings for SupCon Training
    parser.add_argument('--temperature', type=float, default=0.1)
    
    # Dataset setting
    parser.add_argument('--dataset', type=str, default='task1', help = 'task1 | task2')
    parser.add_argument('--data_path', type=str, default='./data/dataset')
    
    # Logging
    parser.add_argument('--ablation', type=bool, default=False)
    parser.add_argument('--project', type=str, default='')
    parser.add_argument('--log', type=str, default='knn_letsur')
    parser.add_argument('--save_model', type=bool, default=True)

    args = parser.parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    ngpus = len(str(args.gpu_ids).split(',')) # torch.cuda.device_count()
    args.ngpus = ngpus

    return args  