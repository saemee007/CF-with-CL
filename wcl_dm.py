import os
import math
import time
from math import sqrt
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp


from data import get_dataset
from data.augmentation import *

from util.meter import *
from util.torch_dist_sum import *
from util.accuracy import accuracy
from util.LARS import LARS
from util.parsing import set_seed
from util import parsing
from tqdm import tqdm

from network.wcl import WCL
from torch.distributed import init_process_group, destroy_process_group

def adjust_learning_rate(args, optimizer, epoch, i, iteration_per_epoch):
    T = epoch * iteration_per_epoch + i
    warmup_iters = args.warmup_epochs * iteration_per_epoch
    total_iters = (args.epochs - args.warmup_epochs) * iteration_per_epoch

    if epoch < args.warmup_epochs:
        lr = args.base_lr * 1.0 * T / warmup_iters
    else:
        T = T - warmup_iters
        lr = 0.5 * args.base_lr * (1 + math.cos(1.0 * T / total_iters * math.pi))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(args, train_loader, model, rank, criterion, optimizer, epoch, iteration_per_epoch, base_lr):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    graph_losses = AverageMeter('graph', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, graph_losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        adjust_learning_rate(args, optimizer, epoch, i, iteration_per_epoch)
        data_time.update(time.time() - end)

        if rank is not None:
            data = data.to(rank, non_blocking=True)

        # compute output
        graph_loss = model(data, rank)
        loss = graph_loss

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        graph_losses.update(graph_loss.item(), data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0 and rank == 0:
            progress.display(i)


def main(rank, args):
    from torch.nn.parallel import DistributedDataParallel
    from util.dist_init import dist_init
    
    # set_seed()
    dist_init(rank, args)    
    
    batch_size = args.batch_size
    args.num_workers = int(args.ngpus) * 4

    if 'imagenet' in args.dataset:
        args.base_lr = 0.075 * sqrt(args.batch_size * int(args.ngpus))
    elif 'cifar' in args.dataset:
        args.base_lr = 0.25 * args.batch_size / 256
    elif 'cf' in args.dataset:
        args.base_lr = 0.25 * args.batch_size / 256
    else:
        raise NotImplementedError("Dataset {} does not exist.".format(args.dataset))

    train_dataset = get_dataset(args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    iteration_per_epoch = train_loader.__len__()
    criterion = nn.CrossEntropyLoss().to(rank)

    model = WCL(dim_input=train_dataset[0].size(0)).to(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    param_dict = {}
    for k, v in model.named_parameters():
        param_dict[k] = v

    bn_params = [v for n, v in param_dict.items() if ('bn' in n or 'bias' in n)]
    rest_params = [v for n, v in param_dict.items() if not ('bn' in n or 'bias' in n)]

    optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0, 'ignore': True },
                                {'params': rest_params, 'weight_decay': 1e-6, 'ignore': False}], 
                                lr=args.base_lr, momentum=0.9, weight_decay=1e-6)

    optimizer = LARS(optimizer, eps=0.0)
    model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    checkpoint_path = 'checkpoints/wcl-{}.pth'.format(args.epochs)
    os.makedirs(checkpoint_path.split('/')[0], exist_ok=True)
    print('checkpoint_path:', checkpoint_path)
    
    if os.path.exists(checkpoint_path):
        checkpoint =  torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(dict(('module.' + k, v) for k, v in checkpoint['model'].items()))
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
    

    model.train()
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        train(args, train_loader, model, rank, criterion, optimizer, epoch, iteration_per_epoch, args.base_lr)
        
        if rank == 0:
            torch.save(
            {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1
            }, checkpoint_path)
    destroy_process_group()

if __name__ == "__main__":
    args = parsing.parse_args()
    world_size = args.ngpus
    mp.spawn(main, args=(args,), nprocs=world_size)