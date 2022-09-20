import json
import torch
from utils import label2idx
from tqdm import tqdm
import numpy as np
import os
import torch.nn.functional as F


def prob_average(path_list, output_path, test_path):
    """
    输出结果为得票最多的label，若同时有多个label票数相同，那么从中选取平均概率最大的那一个作为结果。
    """

    print('loading results files...')
    predictions = []
    for path in path_list:
        predictions.append(torch.load(path))
    
    count = {doc_id: {candidate_id: {'probs': torch.zeros(len(label2idx))} for candidate_id in predictions[0][doc_id]} for doc_id in predictions[0]}
    
    for doc_id in tqdm(count, desc='Probs aggregating'):
        for candidate_id in count[doc_id]:
            for pred in predictions:
                logits = torch.tensor(pred[doc_id][candidate_id]['logits'])
                probs = F.softmax(logits, dim=-1)
                count[doc_id][candidate_id]['probs'] += probs
            probs = count[doc_id][candidate_id].pop('probs')
            idx = torch.argmax(probs)
            count[doc_id][candidate_id]['type_id'] = int(idx)
    
    with open(output_path, 'w', encoding='utf-8') as writer:
         with open(test_path, "r", encoding='utf-8') as fin:
            lines = fin.readlines()
            for line in tqdm(lines, desc='Writing final results'):
                doc = json.loads(line)
                res = {}
                doc_id = doc['id']
                res['id'] = doc_id
                res['predictions'] = []
                for mention in doc['candidates']:
                    mention_id = mention['id']
                    res['predictions'].append({"id": mention_id, "type_id": count[doc_id][mention_id]['type_id']})
                writer.write(json.dumps(res, ensure_ascii=False) + "\n")


def voting(path_list, output_path, test_path, mode='plurality'):
    """
    输出结果为得票最多的label，若同时有多个label票数相同，那么从中选取平均概率最大的那一个作为结果。
    """

    print('loading results files...')
    predictions = []
    for path in path_list:
        predictions.append(torch.load(path))
    
    count = {doc_id: {candidate_id: {'votes': [0]*len(label2idx), 'probs': torch.zeros(len(label2idx))} for candidate_id in predictions[0][doc_id]} for doc_id in predictions[0]}
    
    for doc_id in tqdm(count, desc='Vote Counting'):
        for candidate_id in count[doc_id]:
            for pred in predictions:
                type_id = pred[doc_id][candidate_id]['type_id']
                logits = torch.tensor(pred[doc_id][candidate_id]['logits'])
                probs = F.softmax(logits, dim=-1)
                count[doc_id][candidate_id]['votes'][type_id] += 1
                count[doc_id][candidate_id]['probs'] += probs
            votes = torch.tensor(count[doc_id][candidate_id].pop('votes'))
            probs = count[doc_id][candidate_id].pop('probs')

            idx = torch.where(votes == torch.max(votes))[0]
            if len(idx) != 1:   # 如果有多个label票数相同，则取平均概率最大的label
                idx = idx[torch.argmax(probs[idx])]
            count[doc_id][candidate_id]['type_id'] = int(idx)
    
    with open(output_path, 'w', encoding='utf-8') as writer:
         with open(test_path, "r", encoding='utf-8') as fin:
            lines = fin.readlines()
            for line in tqdm(lines, desc='Writing final results'):
                doc = json.loads(line)
                res = {}
                doc_id = doc['id']
                res['id'] = doc_id
                res['predictions'] = []
                for mention in doc['candidates']:
                    mention_id = mention['id']
                    res['predictions'].append({"id": mention_id, "type_id": count[doc_id][mention_id]['type_id']})
                writer.write(json.dumps(res, ensure_ascii=False) + "\n")


result_dir = './result/bert-base-chinese/FurtherPretrain/epoch5'
path_list = [os.path.join(result_dir, f'fold-{i}', 'logits_stage2') for i in range(5)]
output_path = os.path.join(result_dir, 'probs_ensemble_01234.jsonl')
test_path = './data/test_stage2_corrected.jsonl'
prob_average(path_list, output_path, test_path)
