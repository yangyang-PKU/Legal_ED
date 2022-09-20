import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
from utils import set_seed
import torch.distributed as dist
import torch.multiprocessing as mp
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

 
class RandomDataset(Dataset):
        def __init__(self, rank):
            self.len = 12
            self.data = torch.stack([torch.ones(1), torch.ones(1)*2,torch.ones(1)*3,torch.ones(1)*4,torch.ones(1)*5,torch.ones(1)*6,torch.ones(1)*7,torch.ones(1)*8,torch.ones(1)*9,torch.ones(1)*10,torch.ones(1)*11,torch.ones(1)*12]).to('cuda')
            self.local_rank = rank
        def __getitem__(self, index):
            return self.data[index]
        def __len__(self):
            return self.len

def setup(rank, world_size, seed):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '13500'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    set_seed(seed)   # 这个种子负责除读取数据外的随机性，比如各个进程上的模型参数初始化等


def main(rank, seed, world_size, batch_size_per_gpu, epoch_nums):
    setup(rank, world_size, seed)
    
    dataset = RandomDataset(rank)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False, seed=seed)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size_per_gpu, sampler=sampler, drop_last=False)
    epoch = 0
    while epoch < epoch_nums:
        dist.barrier()
        print("epoch:", epoch)
        dist.barrier()
        sampler.set_epoch(epoch)
        for data in data_loader:
            dist.barrier()
            print(data)
            dist.barrier()
        epoch+=1
 

if __name__ == "__main__":
    seed = 888
    batch_size_per_gpu = 3
    epoch_nums = 3
    world_size = torch.cuda.device_count()    
    mp.spawn(
        main,
        args=(seed, world_size, batch_size_per_gpu, epoch_nums),
        nprocs=world_size
    )
    

    