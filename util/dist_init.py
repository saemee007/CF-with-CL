import os
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def dist_init(rank, args, port=23456):
    
    world_size = args.ngpus
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    
    init_process_group(backend="nccl", rank=rank, world_size=world_size)