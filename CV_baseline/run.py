# coding: UTF-8
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from dataset import TokenClassificationDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from train_eval import train, evaluate, ensemble, pseudo_label
from utils import init_logger, collate_fn, set_seed, metrics, idx2label
from model import TokenClassificationModel, DMBERT
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from config import Config
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
# os.environ["NCCL_DEBUG"] = "INFO"
# os.environ["NCCL_SHM_DISABLE"] = '1'


def setup(rank, world_size, seed):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '13500'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    set_seed(seed)      # 这个种子负责除读取数据外的随机性，比如各个进程上的模型参数初始化等
    torch.cuda.set_device(rank)


def prepare(config, logger, mode, rank, world_size, pin_memory=False, num_workers=0):
    dataset = TokenClassificationDataset(config, logger, mode=mode)
    shuffle = True if mode == 'train' else False
    # 传给sampler的seed要保证各个进程一样，这个seed负责读取数据顺序的随机性
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=False, seed=config.seed)
    dataloader = DataLoader(dataset, 
                            batch_size=config.batch_size_per_gpu,
                            pin_memory=pin_memory,
                            num_workers=num_workers,
                            drop_last=False,
                            shuffle=False,
                            sampler=sampler,
                            collate_fn=lambda x: collate_fn(x, config)
                            )
    return dataloader


def main(rank, config, world_size):
    setup(rank, world_size, config.seed)
    config.device = torch.device("cuda", rank)
    logger = init_logger(config, rank)   # 初始化logger
    if config.do_train & config.do_infer:
        logger.error("Argument 'do_train' and 'do_infer' can't be both True.")
    
    if config.do_train:
        train_dataloader = prepare(config, logger, mode='train', rank=rank, world_size=world_size, pin_memory=False, num_workers=0)
        valid_dataloader = prepare(config, logger, mode='valid', rank=rank, world_size=world_size, pin_memory=False, num_workers=0)
        model = TokenClassificationModel(config).to(config.device)
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        train(config, model, train_dataloader, valid_dataloader, logger)

    elif config.do_infer:
        test_dataloader = prepare(config, logger, mode='test', rank=rank, world_size=world_size, pin_memory=False, num_workers=0)
        model = TokenClassificationModel(config).to(rank)
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)      # 同步每一个进程中的模型参数
        checkpoint_path = "./saved/bert-base-chinese/SET01/fold-3/f1_82.599.pth"
        model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
        # pseudo_label(config, model, test_dataloader, mode='test')
        evaluate(config, model, test_dataloader, mode='test')

        # pred, true, indiv_loss = evaluate(config, model, test_dataloader, mode='valid')
        # collected_pred = [None for _ in range(dist.get_world_size())]
        # collected_true = [None for _ in range(dist.get_world_size())]
        # collected_loss = [None for _ in range(dist.get_world_size())]
        # dist.all_gather_object(collected_pred, pred)
        # dist.all_gather_object(collected_true, true)
        # dist.all_gather_object(collected_loss, indiv_loss)
        # if dist.get_rank() == 0:
        #     collected_pred = [j for i in collected_pred for j in i]
        #     collected_true = [j for i in collected_true for j in i]
        #     dev_loss = sum(collected_loss)/len(collected_loss)
        #     macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, p, r, f1 = metrics(collected_pred, collected_true)
        #     average_f1 = (micro_f1 + macro_f1) / 2
        #     p = {idx2label[i]: p[i-1] for i in range(1, 109)}
        #     r = {idx2label[i]: r[i-1] for i in range(1, 109)}
        #     f1 = {idx2label[i]: f1[i-1] for i in range(1, 109)}
        #     print('micro-f1: ', micro_f1)
        #     print('macro-f1: ', macro_f1)
        #     print('average-f1: ', average_f1)
        #     # print('precision: ', p)
        #     # print('recall: ', r)
        #     # print('f1: ', f1)
    
    elif config.ensemble:
        test_dataloader = prepare(config, logger, mode='test', rank=rank, world_size=world_size, pin_memory=False, num_workers=0)
        checkpoint_paths = ["./saved/bert-base-chinese/SET01/fold-0/f1_82.680.pth",
                            "./saved/bert-base-chinese/SET01/fold-1/f1_82.508.pth",
                            "./saved/bert-base-chinese/SET01/fold-2/f1_82.248.pth",
                            "./saved/bert-base-chinese/SET01/fold-3/f1_82.599.pth",
                            "./saved/bert-base-chinese/SET01/fold-4/f1_82.883.pth",]
        model_list = []
        for checkpoint_path in checkpoint_paths:
            model = TokenClassificationModel(config).to(rank)
            model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)      # 同步每一个进程中的模型参数
            model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
            model_list.append(model)

        ensemble(config, model_list, test_dataloader, mode='test')


if __name__ == '__main__':
    config = Config()
    if not os.path.exists(config.save_dir) and config.do_train:
        os.makedirs(config.save_dir)

    world_size = torch.cuda.device_count()    
    mp.spawn(
        main,
        args=(config, world_size),
        nprocs=world_size
    )