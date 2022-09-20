import torch.distributed as dist
from utils import InputExample, label2idx
import torch
from torch.utils.data import Dataset
import jsonlines
import numpy as np
import json
import os
from tqdm import tqdm


class TokenClassificationDataset(Dataset):
    
    def __init__(self, config, logger, mode):
        self.config = config
        self.mode = mode

        if dist.get_rank()!=0:
            dist.barrier()

        cached_data_file = os.path.join(
        config.data_dir if mode=='test' else config.data_dir + f'/fold_{config.fold}', 
        "cached-{}-{}-{}".format(
        getattr(config, mode+'_path').split('/')[-1].split('.')[0],
        config.model_type,
        f'Truncated_{config.window_size}' if config.truncated else 'None'
        ))

        if os.path.exists(cached_data_file) and not config.overwrite_cache:
            logger.info(f"Loading data from cached file {cached_data_file}")
            self.examples = torch.load(cached_data_file)
        else:
            logger.info(f"Creating cached file at {config.data_dir}")
            
            records = []
            with open(getattr(config, mode+'_path'), 'r') as f:
                lines = f.readlines()
            for line in lines:
                doc = json.loads(line)
                if mode != 'test':
                    for event in doc['events']:
                        for mention in event['mention']:
                            records.append(
                                {   
                                    'doc_id': doc['id'],
                                    'candidate_id': mention['id'],
                                    'tokens': doc['content'][mention['sent_id']]['tokens'],
                                    'offset': [mention['offset'][0], mention['offset'][1]],
                                    'label_id': label2idx[event['type']],
                                }
                            )
                    for nt in doc['negative_triggers']:
                        records.append(
                                {
                                    'doc_id': doc['id'],
                                    'candidate_id': nt['id'],
                                    'tokens': doc['content'][nt['sent_id']]['tokens'],
                                    'offset': [nt['offset'][0], nt['offset'][1]],
                                    'label_id': label2idx['None'],
                                }
                            )
                else:
                    for mention in doc['candidates']:
                        records.append(
                                {
                                    'doc_id': doc['id'],
                                    'candidate_id': mention['id'],
                                    'tokens': doc['content'][mention['sent_id']]['tokens'],
                                    'offset': [mention['offset'][0], mention['offset'][1]],
                                    'label_id': config.pad_label_id,
                                }
                            )

            self.examples = []
            for record in tqdm(records, desc='convert examples to features', disable=(dist.get_rank()!=0)):
                textL = config.tokenizer.tokenize("".join(record['tokens'][:record['offset'][0]]))
                textL += ['[unused0]']
                textCandidate = config.tokenizer.tokenize("".join(record['tokens'][record['offset'][0]: record['offset'][1]]))
                textR = ['[unused1]']
                textR += config.tokenizer.tokenize("".join(record['tokens'][record['offset'][1]:]))

                if not config.truncated:
                    left_idx = len(textL)
                    right_idx = left_idx + len(textCandidate)
                    candidate_mask = [0]*len(textL) + [1] + [0]*(len(textCandidate)-1) + [0]*len(textR)   # 只对candidate第一个token标注
                    label_id = [config.pad_label_id]*len(textL) + [record['label_id']] + [config.pad_label_id]*(len(textCandidate)-1) + [config.pad_label_id]*len(textR)
                    tokens = textL + textCandidate + textR
                else:
                    left_idx = min(len(textL), config.window_size)
                    right_idx = left_idx + len(textCandidate)
                    candidate_mask = [0]*min(len(textL), config.window_size) + [1] + [0]*(len(textCandidate)-1) + [0]*min(len(textR), config.window_size)
                    label_id = [config.pad_label_id]*min(len(textL), config.window_size) + [record['label_id']] + [config.pad_label_id]*(len(textCandidate)-1) + [config.pad_label_id]*min(len(textR), config.window_size)
                    tokens = textL[len(textL) - min(len(textL), config.window_size):] + textCandidate + textR[: min(len(textR), config.window_size)]

                assert tokens[left_idx - 1]=='[unused0]'
                assert tokens[right_idx]=='[unused1]'
                assert len(tokens)==len(candidate_mask)==len(label_id)

                self.examples.append(
                                {
                                    'doc_id': record['doc_id'],
                                    'candidate_id': record['candidate_id'],
                                    'tokens': tokens,
                                    'offset': [left_idx, right_idx],
                                    'label_id': label_id,
                                    'candidate_mask': candidate_mask,
                                    'truncated': False,     # 是否因超长而导致candidate位置被截去
                                }
                            )
            logger.info(f"Saving data into cached file {cached_data_file}")
            torch.save(self.examples, cached_data_file)

        if dist.get_rank()==0:
            dist.barrier()

    def __getitem__(self, idx):
        return self.examples[idx]
        

    def __len__(self):
        return len(self.examples)


class DMDataset(Dataset):
    
    def __init__(self, config, logger, mode):
        self.config = config
        self.mode = mode

        if dist.get_rank()!=0:
            dist.barrier()

        cached_data_file = os.path.join(
        config.data_dir if mode=='test' else config.data_dir + f'/fold_{config.fold}', 
        "cached_{}_{}_DM".format(
        getattr(config, mode+'_path').split('/')[-1].split('.')[0],
        config.model_type
        ))

        if os.path.exists(cached_data_file) and not config.overwrite_cache:
            logger.info(f"Loading data from cached file {cached_data_file}")
            self.examples = torch.load(cached_data_file)
        else:
            logger.info(f"Creating cached file at {config.data_dir}")
            
            records = []
            with open(getattr(config, mode+'_path'), 'r') as f:
                lines = f.readlines()
            for line in lines:
                doc = json.loads(line)
                if mode != 'test':
                    for event in doc['events']:
                        for mention in event['mention']:
                            records.append(
                                {   
                                    'doc_id': doc['id'],
                                    'candidate_id': mention['id'],
                                    'tokens': doc['content'][mention['sent_id']]['tokens'],
                                    'offset': [mention['offset'][0], mention['offset'][1]],
                                    'label_id': label2idx[event['type']],
                                }
                            )
                    for nt in doc['negative_triggers']:
                        records.append(
                                {
                                    'doc_id': doc['id'],
                                    'candidate_id': nt['id'],
                                    'tokens': doc['content'][nt['sent_id']]['tokens'],
                                    'offset': [nt['offset'][0], nt['offset'][1]],
                                    'label_id': label2idx['None'],
                                }
                            )
                else:
                    for mention in doc['candidates']:
                        records.append(
                                {
                                    'doc_id': doc['id'],
                                    'candidate_id': mention['id'],
                                    'tokens': doc['content'][mention['sent_id']]['tokens'],
                                    'offset': [mention['offset'][0], mention['offset'][1]],
                                    'label_id': config.pad_label_id,
                                }
                            )

            self.examples = []
            for record in tqdm(records, desc='convert examples to features', disable=(dist.get_rank()!=0)):
                textL = config.tokenizer.tokenize("".join(record['tokens'][:record['offset'][0]]))
                textL += ['[unused0]']
                textCandidate = config.tokenizer.tokenize("".join(record['tokens'][record['offset'][0]: record['offset'][1]]))
                textR = ['[unused1]']
                textR += config.tokenizer.tokenize("".join(record['tokens'][record['offset'][1]:]))

                maskL = [1]*(len(textL)-1) + [0]*(1 + len(textCandidate) + len(textR))
                maskR = [0]*(len(textL)-1) + [1]*(1 + len(textCandidate) + len(textR))
                candidate_mask = [0]*len(textL) + [1] + [0]*(len(textCandidate)-1) + [0]*len(textR)   # 只对candidate第一个token标注
                label_id = [config.pad_label_id]*len(textL) + [record['label_id']] + [config.pad_label_id]*(len(textCandidate)-1) + [config.pad_label_id]*len(textR)
                
                tokens = textL + textCandidate + textR
                assert len(tokens)==len(candidate_mask)==len(label_id)==len(maskL)==len(maskR)

                self.examples.append(
                                {
                                    'doc_id': record['doc_id'],
                                    'candidate_id': record['candidate_id'],
                                    'tokens': tokens,
                                    'offset': [len(textL), len(textL) + len(textCandidate)],
                                    'label_id': label_id,
                                    'candidate_mask': candidate_mask,
                                    'maskL': maskL,
                                    'maskR': maskR,
                                    'truncated': False,     # 是否因超长而导致candidate位置被截去
                                }
                            )
            logger.info(f"Saving data into cached file {cached_data_file}")
            torch.save(self.examples, cached_data_file)

        if dist.get_rank()==0:
            dist.barrier()

    def __getitem__(self, idx):
        return self.examples[idx]
        

    def __len__(self):
        return len(self.examples)