# coding: UTF-8
import os
import random
import torch
import numpy as np
import pickle as pkl
import logging
import time
import json
import jsonlines
import math
from copy import deepcopy
from datetime import timedelta
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import precision_score, recall_score, f1_score
from pathlib import Path
from data_process.const import LABELS2IDX


def init_logger(config, rank):
    '''
    Example:
        >>> init_logger(log_file)
        >>> logger.info("abc'")
    '''

    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s -   %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO if rank == 0 else logging.WARN)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if config.if_log_file and config.do_train and rank==0:
        time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())  # 记录当前时间
        file_handler = logging.FileHandler(os.path.join(config.save_dir, 'info.log'),
                                           mode='a')  # mod参数和open函数传入的'a','r','w'等一样
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def set_seed(seed):
    # random.seed(seed)     # 下采样negative trigger需要
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class InputExample(object):
    """
        A single training/test example for token classification.
        one single sequence of tokens is an example in dataset.
    """

    def __init__(self, guid, sentence, tokens, candidates, candidate_mask, token_type_ids, label_ids=None):
        self.guid = guid
        self.sentence = sentence
        self.tokens = tokens
        self.label_ids = label_ids
        self.candidates = candidates
        self.candidate_mask = candidate_mask
        self.token_type_ids = token_type_ids


def build_dict(labels):
    all_labels = list(labels.keys())

    label2idx = {tag: idx for idx, tag in enumerate(all_labels)}
    idx2label = {idx: tag for idx, tag in enumerate(all_labels)}

    return all_labels, label2idx, idx2label

all_labels, label2idx, idx2label = build_dict(LABELS2IDX)


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0., 0.5 * (1. + math.cos(math.pi * float(num_cycles) * 2. * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def metrics(pred, true):
    target_labels = list(idx2label.keys())
    target_labels.remove(0)
    macro_p = precision_score(true, pred, labels=target_labels , average='macro') * 100.0
    macro_r = recall_score(true, pred, labels=target_labels , average='macro') * 100.0
    macro_f1 = f1_score(true, pred, labels=target_labels , average='macro') * 100.0

    p = precision_score(true, pred, labels=target_labels , average=None) * 100.0
    r = recall_score(true, pred, labels=target_labels , average=None) * 100.0
    f1 = f1_score(true, pred, labels=target_labels , average=None) * 100.0

    micro_p = precision_score(true, pred, labels=target_labels , average='micro') * 100.0
    micro_r = recall_score(true, pred, labels=target_labels , average='micro') * 100.0
    micro_f1 = f1_score(true, pred, labels=target_labels , average='micro') * 100.0

    return macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, p, r, f1


def collate_fn(batch, config):
    output = {'input_ids':[], 'attention_mask':[], 'candidate_mask':[], 'token_type_ids':[], 'tokens':[], 'label_ids':[], 'sentence':[], 'candidates':[], 'guids':[]}
    batch = deepcopy(batch)

    special_tokens_count = 2        # [CLS]和[SEP]
    batch_max_len = max([len(example.tokens) for example in batch])
    if batch_max_len > config.max_seq_len - special_tokens_count:
        max_char_len = config.max_seq_len

        for example in batch:
            if len(example.tokens) > config.max_seq_len - special_tokens_count:
                example.tokens = example.tokens[: config.max_seq_len - special_tokens_count]
                example.label_ids = example.label_ids[: config.max_seq_len - special_tokens_count]
                example.candidate_mask = example.candidate_mask[: config.max_seq_len - special_tokens_count]
                example.token_type_ids = example.token_type_ids[: config.max_seq_len - special_tokens_count]
                for candidate in example.candidates:
                    if candidate['offset'][0] < config.max_seq_len - special_tokens_count and candidate['offset'][1] > config.max_seq_len - special_tokens_count:
                        candidate['offset'][1] = config.max_seq_len - special_tokens_count      # 将offset刚好卡在max_seq_len的candidate的offset进行修改
                    elif candidate['offset'][0] >= config.max_seq_len - special_tokens_count:
                        candidate['reduced'] = True     # True表示该candidate由于超过max_seq_len的限制，被截去
            
    else:
        max_char_len = batch_max_len + special_tokens_count

   
    for example in batch:

        example.label_ids = [config.pad_label_id] + example.label_ids + [config.pad_label_id]
        example.candidate_mask = [0] + example.candidate_mask + [0]
        if config.model_type == 'bert':
            example.tokens = ['[CLS]'] + example.tokens + ['[SEP]']
        elif config.model_type == 'roberta':
            example.tokens = ['<s>'] + example.tokens + ['</s>']
        
        input_ids = config.tokenizer.convert_tokens_to_ids(example.tokens)
        attention_mask = [1] * len(input_ids)
        example.token_type_ids = [0] + example.token_type_ids + [1 if config.title else 0]
        
        # 因为句子最前面加了[CLS] token，所以所有candidate的offset向右平移一位
        for candidate in example.candidates:
            candidate['offset'][0] += 1
            candidate['offset'][1] += 1

        assert len(input_ids)==len(attention_mask)==len(example.label_ids)==len(example.candidate_mask)==len(example.token_type_ids)
        if len(input_ids) < max_char_len:
            example.label_ids += ([config.pad_label_id] * (max_char_len - len(example.label_ids)))
            example.candidate_mask += ([0] * (max_char_len - len(example.candidate_mask)))
            example.token_type_ids += ([0] * (max_char_len - len(example.token_type_ids)))
            # mask += [0] * (max_char_len - len(mask))
            attention_mask += [0] * (max_char_len - len(attention_mask))
            input_ids += ([0 if config.model_type == 'bert' else 1] * (max_char_len - len(input_ids)))         # bert-base-chinese的pad token id是0, xlm-roberta的pad token id是1
        assert len(input_ids)==len(attention_mask)==len(example.label_ids)==len(example.token_type_ids)==len(example.candidate_mask)

        output['label_ids'].append(example.label_ids)
        output['input_ids'].append(input_ids)
        output['candidate_mask'].append(example.candidate_mask)
        output['token_type_ids'].append(example.token_type_ids)
        output['attention_mask'].append(attention_mask)
        output['sentence'].append(example.sentence)
        output['candidates'].append(example.candidates)
        output['guids'].append(example.guid)
        output['tokens'].append(example.tokens)

    return output