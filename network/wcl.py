# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import sys

import torch
import torch.nn as nn

sys.path.append('/home/saemeechoi/cls_noise/with_WCL/')
from network.head import *
from network.resnet import *
import torch.nn.functional as F
import torch.nn as nn
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np

class WCL(nn.Module):
    def __init__(self, dim_input, dim_hidden=4096, dim_output=256):
        super(WCL, self).__init__()
        self.net = nn.Linear(dim_input, 2048) # resnet50(pretrained=True)
        self.head = ProjectionHead(dim_in=2048, dim_out=dim_output, dim_hidden=dim_hidden)

    @torch.no_grad()
    def build_connected_component(self, dist):
        b = dist.size(0)
        dist = dist - torch.eye(b, b, device='cuda') * 2
        x = torch.arange(b, device='cuda').unsqueeze(1).repeat(1,1).flatten()
        y = torch.topk(dist, 1, dim=1, sorted=False)[1].flatten()
        rx = torch.cat([x, y]).cpu().numpy()
        ry = torch.cat([y, x]).cpu().numpy()
        v = np.ones(rx.shape[0])
        graph = csr_matrix((v, (rx, ry)), shape=(b,b))
        _, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        labels = torch.tensor(labels, device='cuda')
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T)
        return mask

    def sup_contra(self, logits, mask, diagnal_mask=None):
        if diagnal_mask is not None:
            diagnal_mask = 1 - diagnal_mask
            mask = mask * diagnal_mask
            exp_logits = torch.exp(logits) * diagnal_mask
        else:
            exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = (-mean_log_prob_pos).mean()
        return loss

    def forward(self, x, rank, t=0.1):
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        b = x.size(0)
        bakcbone_feat = self.net(x)
        feat = F.normalize(self.head(bakcbone_feat))
        all_feat = concat_all_gather(feat)
        all_bs = all_feat.size(0)

        mask_list = []
        if rank == 0:
            mask = self.build_connected_component(all_feat @ all_feat.T).float()
            mask_list = list(torch.chunk(mask, world_size))
            mask = mask_list[0]
        else:
            mask = torch.zeros(b, all_bs, device='cuda')
        torch.distributed.scatter(mask, mask_list, 0)

        diagnal_mask = torch.eye(all_bs, all_bs, device='cuda')
        diagnal_mask = torch.chunk(diagnal_mask, world_size)[rank]
        graph_loss =  self.sup_contra(feat @ all_feat.T / t, mask, diagnal_mask)
        
        return graph_loss



# utils
@torch.no_grad()
def concat_other_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    rank = torch.distributed.get_rank()
    tensors_gather = torch.stack([torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())])
    torch.distributed.all_gather_into_tensor(tensors_gather, tensor)
    try:
        other = torch.cat(tensors_gather[:rank] + tensors_gather[rank+1:], dim=0)
    except:
        other = tensors_gather[:rank]
        
    return other



@torch.no_grad()
def concat_all_gather(tensor, replace=True):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    rank = torch.distributed.get_rank()
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)
    if replace:
        tensors_gather[rank] = tensor
    other = torch.cat(tensors_gather, dim=0)
    return other


