from ast import Raise
from logging import raiseExceptions
import torch.distributed as dist
from utils import InputExample, label2idx
import torch
from torch.utils.data import Dataset
import jsonlines
import numpy as np
import json
import os
import random


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
        'Title' if config.title else 'None'
        ))

        if os.path.exists(cached_data_file) and not config.overwrite_cache:
            logger.info(f"Loading data from cached file {cached_data_file}")
            self.examples = torch.load(cached_data_file)
        else:
            logger.info(f"Creating cached file at {config.data_dir}")
            self.examples = []
            with open(getattr(config, mode+'_path'), 'r') as f:
                lines = f.readlines()
            for idx, line in enumerate(lines):
                doc = json.loads(line)
                if idx % 1000 == 0:
                    logger.info(f'{idx}/{len(lines)}')

                title = doc['title']
                sentences = [i['sentence'] for i in doc['content']]
                words = [i['tokens'] for i in doc['content']]
                candidates = [[] for _ in doc['content']]
                token_type_ids = []
                tokens = []
                label_ids = []
                candidate_mask = []
                word_offset = []

                if mode == 'train':
                    for event in doc['events']:
                        for mention in event['mention']:
                            candidates[mention['sent_id']].append({'id': mention['id'], 'offset': mention['offset'], 'type_id': label2idx[event['type']], 'reduced': False, "trigger_word": mention['trigger_word']})

                    # 对于负样本，只采样一半
                    if config.ds_neg:
                        sampled_negative_tri = random.sample(doc['negative_triggers'], len(doc['negative_triggers'])//2)
                        for mention in sampled_negative_tri:
                            candidates[mention['sent_id']].append({'id': mention['id'], 'offset': mention['offset'], 'type_id': 0, 'reduced': False, "trigger_word": mention['trigger_word']})
                    else:
                        for mention in doc['negative_triggers']:
                            candidates[mention['sent_id']].append({'id': mention['id'], 'offset': mention['offset'], 'type_id': 0, 'reduced': False, "trigger_word": mention['trigger_word']})
                
               
                elif mode == 'valid':
                    for event in doc['events']:
                        for mention in event['mention']:
                            candidates[mention['sent_id']].append({'id': mention['id'], 'offset': mention['offset'], 'type_id': label2idx[event['type']], 'reduced': False, "trigger_word": mention['trigger_word']})

                    for mention in doc['negative_triggers']:
                        candidates[mention['sent_id']].append({'id': mention['id'], 'offset': mention['offset'], 'type_id': 0, 'reduced': False, "trigger_word": mention['trigger_word']})


                else:   # inference (test)
                    for mention in doc['candidates']:
                        candidates[mention['sent_id']].append({'id': mention['id'], 'offset': mention['offset'], 'type_id': -1, 'reduced': False, "trigger_word": mention['trigger_word']})     # test时先初始化所有candidate的预测结果都是-1

                for i in range(0, len(words)):
                    # 一句话一句话的处理，每一句话是一个example，这里i相当于sent_id
                    sentences[i] += '。'
                    words[i].append('。')
                    
                    cur_tokens = []
                    cur_label_ids = []
                    cur_token_type_ids = []
                    cur_candidate_mask = []
                    cur_word_offset = []
                    cursor = 0

                    if config.title:
                        word_tokens = self.config.tokenizer.tokenize(title)
                        if len(word_tokens) == 0:
                            word_tokens = ['[UNK]'] if config.model_type == 'bert' else ['<unk>']
                        word_tokens.append('[SEP]' if config.model_type == 'bert' else '</s>')
                        cur_tokens.extend(word_tokens)
                        # cur_word_offset.append([cursor, cursor + len(word_tokens)])
                        cursor += len(word_tokens)
                        cur_token_type_ids.extend([0] * len(word_tokens))
                        cur_label_ids.extend([config.pad_label_id] * len(word_tokens))
                        cur_candidate_mask.extend([0] * len(word_tokens))
                    for word in words[i]:
                        # 这里是一个word一个word的处理
                        word_tokens = self.config.tokenizer.tokenize(word)
                        if len(word_tokens) == 0:
                            word_tokens = ['[UNK]'] if config.model_type == 'bert' else ['<unk>']
                        cur_tokens.extend(word_tokens)
                        cur_word_offset.append([cursor, cursor + len(word_tokens)])
                        cursor += len(word_tokens)
                        cur_token_type_ids.extend([1 if config.title else 0] * len(word_tokens))
                        cur_label_ids.extend([config.pad_label_id] * len(word_tokens))
                        cur_candidate_mask.extend([0] * len(word_tokens)) 
                    
                    assert len(cur_tokens)==len(cur_label_ids)==len(cur_candidate_mask)==len(cur_token_type_ids)

                    for candidate in candidates[i]:
                        # 将candidate中的offset由word level变成token level
                        start_token_idx = cur_word_offset[candidate['offset'][0]][0]
                        end_token_idx = cur_word_offset[candidate['offset'][1] - 1][1]
                        candidate['offset'][0] = start_token_idx
                        candidate['offset'][1] = end_token_idx

                        # 标注label_ids和candidate_mask，只对一个candidate的第一个token（字）标注
                        cur_label_ids[start_token_idx] = candidate['type_id']
                        cur_candidate_mask[start_token_idx] = 1
                    
                    tokens.append(cur_tokens)
                    label_ids.append(cur_label_ids)
                    candidate_mask.append(cur_candidate_mask)
                    word_offset.append(cur_word_offset)
                    token_type_ids.append(cur_token_type_ids)


                for i in range(len(words)):
                    cur_tokens = tokens[i]
                    cur_sentence = sentences[i]
                    cur_label_ids = label_ids[i]
                    cur_candidate_mask = candidate_mask[i]
                    cur_token_type_ids = token_type_ids[i]
                    
                    cur_candidates = candidates[i]
                    if len(cur_candidates) == 0:            # 如果当前句子里没有任何candidate出现，则不加入到dataset中
                        continue

                    if config.context:# 加入context
                        if config.title:
                            raise ValueError("pre-context 和 title 不能同时为True")
                        if i != 0:
                            # 如果当前句子的长度小于100则加上前一句作为context
                            if len(cur_tokens) < 100 and len(tokens[i - 1]) + len(cur_tokens) <= 510:
                                cur_sentence = sentences[i - 1] + cur_sentence
                                cur_tokens = tokens[i - 1] + cur_tokens
                                cur_label_ids = [config.pad_label_id] * len(tokens[i - 1]) + cur_label_ids
                                cur_candidate_mask = [0] * len(tokens[i - 1]) + cur_candidate_mask
                                for candidate in candidates[i]:
                                    candidate['offset'][0] += len(tokens[i - 1])
                                    candidate['offset'][1] += len(tokens[i - 1])
                        # if i != (len(words) - 1):
                        #     # 加上后一句作为context
                        #     if len(tokens[i + 1]) + len(cur_tokens) <= 510:
                        #         cur_sentence = cur_sentence + sentences[i + 1]
                        #         cur_tokens = cur_tokens + tokens[i + 1]
                        #         cur_label_ids = cur_label_ids + [config.pad_label_id] * len(tokens[i + 1])
                        #         cur_candidate_mask = cur_candidate_mask + [0] * len(tokens[i + 1])

                    self.examples.append(InputExample(guid="%s-%d" % (doc['id'], i),
                                                    sentence=cur_sentence,
                                                    tokens=cur_tokens,
                                                    label_ids=cur_label_ids,
                                                    candidates=sorted(candidates[i], key=lambda x: x['offset'][0]),
                                                    candidate_mask=cur_candidate_mask,
                                                    token_type_ids=cur_token_type_ids))
            logger.info(f"Saving data into cached file {cached_data_file}")
            torch.save(self.examples, cached_data_file)

        if dist.get_rank()==0:
            dist.barrier()

    def __getitem__(self, idx):
        return self.examples[idx]
        

    def __len__(self):
        return len(self.examples)