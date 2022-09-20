import torch
import json
from transformers import AdamW
from utils import get_cosine_schedule_with_warmup
from tqdm import tqdm
from data_process.const import hard_label_idx
import torch.nn as nn
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import torch.nn.functional as F
from utils import metrics


def train(config, model, train_dataloader, valid_dataloader, logger):
    batch_size = torch.distributed.get_world_size() * config.batch_size_per_gpu
    total_step = len(train_dataloader) // config.gradient_accumulation_steps * config.num_epochs
    warmup_steps = int(total_step * config.warmup_proportion)
    not_eval_until = int(config.not_eval_until * len(train_dataloader))

    bert_parameters = model.module.bert.named_parameters()
    classifier_parameters = model.module.linear.named_parameters()
    no_decay = ['bias', 'LayerNorm.weight']  # 不需要weight_decay的参数
    optimizer_grouped_parameters = [
        {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": config.weight_decay, 'lr': config.encoder_learning_rate},
        {"params": [p for n, p in bert_parameters if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': config.encoder_learning_rate},
        {"params": [p for n, p in classifier_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": config.weight_decay, 'lr': config.classifier_learning_rate},
        {"params": [p for n, p in classifier_parameters if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': config.classifier_learning_rate},
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr)
    # optimizer = AdamW(model.parameters(), lr=config.lr)
    if config.warmup:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_step)
    
    
    logger.info('\n')
    logger.info("***** Running training *****")
    logger.info("  Train path = {}, Valid path = {}".format(config.train_path, config.valid_path))
    logger.info("  Num examples = %d", len(train_dataloader.dataset))
    logger.info("  Num epochs = %d", config.num_epochs)
    # logger.info("  Batch size = %d", config.batch_size)
    logger.info("  Max length = %d", config.max_seq_len)
    logger.info("  Total optimization steps = %d", total_step)
    logger.info("  Warmup proportion = {}, Total warmup steps = {}".format(config.warmup_proportion, warmup_steps))
    logger.info("  Total train batch size (w. DDP ) = %d", batch_size)
    logger.info("  Gradient Accumulation steps = %d", config.gradient_accumulation_steps)
    logger.info("  Dropout rate = {}".format(config.dropout))
    logger.info("  Weight decay = {}".format(config.weight_decay))
    logger.info("  Loss Weights = {}".format(str(config.weights)))
    logger.info("  With Pre-Context = {}".format('Crime' if config.crime else 'None'))
    logger.info("  Down Sampling Negative Triggers = {}".format(str(config.ds_neg)))
    logger.info("  learning rate of encoder = {}, learning rate of classifier = {}".format(
        config.encoder_learning_rate, config.classifier_learning_rate))
    logger.info("  Max_grad_norm = {}".format(config.max_grad_norm))
    logger.info("  Model name = {}".format(config.model_name_or_path))
    logger.info("  Require improvement steps = {}".format(config.require_improvement))
    logger.info("  Improvement metric = {}".format(config.improvement_metric))
    logger.info("  Eval step = {}".format(config.eval_step))
    logger.info("  Not eval until = {}".format(config.not_eval_until))
    logger.info("  Seed = {}, Sampler Seed = {}".format(config.seed, config.seed))

    logger.info("\n")


    current_step = 1
    dev_best_loss = float('inf')
    dev_best_f1 = 0.0
    last_improve = not_eval_until  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    for epoch in range(config.num_epochs):
        train_bar = tqdm(train_dataloader, desc="Training", disable=(dist.get_rank()!=0))
        train_dataloader.sampler.set_epoch(epoch)
        for data in train_bar:
            input_ids = data['input_ids']
            token_type_ids = data['token_type_ids']
            attention_mask = data['attention_mask']
            candidate_mask = data['candidate_mask']
            label_ids = data['label_ids']

            input_ids = torch.tensor(input_ids, dtype=torch.int64, device=config.device)              # (batch_size, pad_size)
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.int64, device=config.device)    # (batch_size, pad_size)
            attention_mask = torch.tensor(attention_mask, dtype=torch.bool, device=config.device)     # (batch_size, pad_size)
            candidate_mask = torch.tensor(candidate_mask, dtype=torch.bool, device=config.device)     # (batch_size, pad_size)
            label_ids = torch.tensor(label_ids, dtype=torch.int64, device=config.device)

            loss, _ = model(input_ids, attention_mask, token_type_ids, candidate_mask, label_ids)
            # loss = loss.mean()

            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps
            
            loss.backward()

            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            if current_step % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                if config.warmup:
                    scheduler.step()
            
            
            train_bar.set_description("epoch:{}/{} - batch:{}, loss:{:.6f} ".format(epoch + 1, int(config.num_epochs),
                                                                                   current_step, loss.item()))

            # evaluate and print metrics
            if current_step > not_eval_until and current_step % config.eval_step == 0:
                pred, true, indiv_loss = evaluate(config, model, valid_dataloader, mode='valid')
                collected_pred = [None for _ in range(dist.get_world_size())]
                collected_true = [None for _ in range(dist.get_world_size())]
                collected_loss = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(collected_pred, pred)
                dist.all_gather_object(collected_true, true)
                dist.all_gather_object(collected_loss, indiv_loss)

                if dist.get_rank() == 0:
                    collected_pred = [j for i in collected_pred for j in i]
                    collected_true = [j for i in collected_true for j in i]
                    dev_loss = sum(collected_loss)/len(collected_loss)
                    
                    macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, _, _, _ = metrics(collected_pred, collected_true)
                    
                    average_f1 = (micro_f1 + macro_f1) / 2

                    if config.improvement_metric == "loss":
                        if dev_loss < dev_best_loss:
                            dev_best_loss = dev_loss
                            improve = '*'
                            last_improve = current_step
                            torch.save(model.state_dict(),
                                    os.path.join(config.save_dir,
                                    config.improvement_metric + f"_{dev_loss:.5f}" + f'_f1_{average_f1:.4f}' ".pth"))
                        else:
                            improve = ''
                        logger.info('epoch%d, step%5d | dev loss: %.5f | dev best loss: %.5f |average_f1: %.3f | %s' % (
                                    epoch + 1, current_step, dev_loss, dev_best_loss, average_f1, improve))
                    else:
                        if average_f1 > dev_best_f1:
                            dev_best_f1 = average_f1
                            improve = '*'
                            last_improve = current_step
                            torch.save(model.state_dict(),
                                    os.path.join(config.save_dir, config.improvement_metric + f"_{average_f1:.3f}"  + ".pth"))
                        else:
                            improve = ''
                        logger.info('epoch%d, step%5d | dev loss: %.5f | best F1: %.3f | avg f1: %.3f | micro f1: %.3f | macro f1: %.3f | %s' % (
                                    epoch + 1, current_step, dev_loss, dev_best_f1, average_f1, micro_f1, macro_f1, improve))
                
                collected_improve = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(collected_improve, last_improve)
                last_improve = max(collected_improve)
                model.train()

            current_step += 1
            if current_step - last_improve > config.require_improvement:
                # 验证集loss超过一定steps还没下降，则结束训练
                logger.info("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    dist.destroy_process_group()
    

def evaluate(config, model, eval_dataloader, mode):
    model.eval()
    total_loss = 0.0
    results = {}
    true = []
    pred = []
    with torch.no_grad():
        for data in tqdm(eval_dataloader, desc="Evaluating", disable=(dist.get_rank()!=0)):
            input_ids = data['input_ids']
            token_type_ids = data['token_type_ids']
            attention_mask = data['attention_mask']
            candidate_mask = data['candidate_mask']
            label_ids = data['label_ids']
            candidates = data['candidates']
            guids = data['guids']
            input_ids = torch.tensor(input_ids, dtype=torch.int64, device=config.device)            # (batch_size, pad_size)
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.int64, device=config.device)  # (batch_size, pad_size)
            attention_mask = torch.tensor(attention_mask, dtype=torch.bool, device=config.device)   # (batch_size, pad_size)
            candidate_mask = torch.tensor(candidate_mask, dtype=torch.bool, device=config.device)   # (batch_size, pad_size)
            label_ids = torch.tensor(label_ids, dtype=torch.int64, device=config.device)
            
            if mode=='valid':   # valid
                loss, logits = model(input_ids, attention_mask, token_type_ids, candidate_mask, label_ids)    # logits: (all_candidates_nums_in_batch, lable_nums)
                total_loss += loss
                prediction = torch.argmax(logits, dim=-1).tolist()          # (all_candidates_nums_in_batch, lable_nums)
                label_ids = label_ids[candidate_mask]           # (all_candidates_nums_in_batch, lable_nums)
                label_ids = label_ids.tolist()
                true += label_ids
                pred += prediction
            
            else:  # test
                logits = model(input_ids, attention_mask, token_type_ids, candidate_mask)    # (all_candidates_nums_in_batch, lable_nums)
                prediction = torch.argmax(logits, dim=-1).tolist()                           # (all_candidates_nums_in_batch,)

                # 将prediction重新填充为(batch_size, pad_size)的形状
                pred = [[] for _ in guids]
                cnt = 0
                for i, mask in enumerate(candidate_mask):
                    for m in mask:
                        if m==0:
                            pred[i].append(config.pad_label_id)
                        else:
                            pred[i].append(prediction[cnt])
                            cnt += 1
                assert cnt==len(prediction)


                for idx in range(len(guids)):   # idx是一个batch中句子的索引
                    doc_id = guids[idx].split('-')[0]
                    if doc_id not in results:
                        results[doc_id] = {}

                    for candidate in candidates[idx]:
                        if not candidate['reduced']:
                            assert pred[idx][candidate['offset'][0]] != config.pad_label_id
                            results[doc_id][candidate['id']] = pred[idx][candidate['offset'][0]]    # 取每个candidate第一个token的预测结果作为该candidate的结果
                        else:
                            results[doc_id][candidate['id']] = 0  # 由于max_seq_len被截断的candidate预测为None类


    if mode=='valid':
        return pred, true, total_loss.cpu()/len(eval_dataloader)
    
    else:
        # 输出预测结果
        collected_results = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(collected_results, results)
        
        if dist.get_rank() == 0:
            results = {}
            for result in collected_results:
                for doc_id, doc_dict in result.items():
                    if doc_id not in results:
                        results[doc_id] = {}
                    for candidate_id, type_id in doc_dict.items():
                        results[doc_id][candidate_id] = type_id


            output_path = os.path.join(config.output_dir, 'result_' + config.test_path.split('/')[-1].split('_')[1] + '.jsonl')
            with open(output_path, "w", encoding="utf8") as writer:
                with open(config.test_path, "r") as fin:
                    lines = fin.readlines()
                    for line in lines:
                        doc = json.loads(line)
                        res = {}
                        doc_id = doc['id']
                        res['id'] = doc_id
                        res['predictions'] = []
                        for mention in doc['candidates']:
                            mention_id = mention['id']
                            res['predictions'].append({"id": mention_id, "type_id": results[doc_id][mention_id]})
                        writer.write(json.dumps(res, ensure_ascii=False) + "\n")
        dist.destroy_process_group()


def ensemble(config, model_list, eval_dataloader, mode):
    for model in model_list:
        model.eval()
    total_loss = 0.0
    results = {}
    true = []
    pred = []
    with torch.no_grad():
        for data in tqdm(eval_dataloader, desc="Evaluating", disable=(dist.get_rank()!=0)):
            input_ids = data['input_ids']
            token_type_ids = data['token_type_ids']
            attention_mask = data['attention_mask']
            candidate_mask = data['candidate_mask']
            label_ids = data['label_ids']
            candidates = data['candidates']
            guids = data['guids']
            input_ids = torch.tensor(input_ids, dtype=torch.int64, device=config.device)            # (batch_size, pad_size)
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.int64, device=config.device)  # (batch_size, pad_size)
            attention_mask = torch.tensor(attention_mask, dtype=torch.bool, device=config.device)   # (batch_size, pad_size)
            candidate_mask = torch.tensor(candidate_mask, dtype=torch.bool, device=config.device)   # (batch_size, pad_size)
            label_ids = torch.tensor(label_ids, dtype=torch.int64, device=config.device)
            
            probs = None
            for model in model_list:
                logits = model(input_ids, attention_mask, token_type_ids, candidate_mask)    # (all_candidates_nums_in_batch, lable_nums)
                if probs is None:
                    probs = F.softmax(logits, dim=-1).unsqueeze(0)                           # (1, all_candidates_nums_in_batch, lable_nums)
                else:
                    probs = torch.cat((probs, F.softmax(logits, dim=-1).unsqueeze(0)), dim=0)   # (model_nums, all_candidates_nums_in_batch, lable_nums)
            probs = torch.mean(probs, dim=0)        # (all_candidates_nums_in_batch, lable_nums)
            prediction = torch.argmax(probs, dim=-1).tolist()   # (all_candidates_nums_in_batch,)
            
            
            # 将prediction重新填充为(batch_size, pad_size)的形状
            pred = [[] for _ in guids]
            cnt = 0
            for i, mask in enumerate(candidate_mask):
                for m in mask:
                    if m==0:
                        pred[i].append(config.pad_label_id)
                    else:
                        pred[i].append(prediction[cnt])
                        cnt += 1
            assert cnt==len(prediction)


            for idx in range(len(guids)):   # idx是一个batch中句子的索引
                doc_id = guids[idx].split('-')[0]
                if doc_id not in results:
                    results[doc_id] = {}

                for candidate in candidates[idx]:
                    if not candidate['reduced']:
                        assert pred[idx][candidate['offset'][0]] != config.pad_label_id
                        results[doc_id][candidate['id']] = pred[idx][candidate['offset'][0]]    # 取每个candidate第一个token的预测结果作为该candidate的结果
                    else:
                        results[doc_id][candidate['id']] = 0  # 由于max_seq_len被截断的candidate预测为None类

    # 输出预测结果
    collected_results = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(collected_results, results)
    
    if dist.get_rank() == 0:
        results = {}
        for result in collected_results:
            for doc_id, doc_dict in result.items():
                if doc_id not in results:
                    results[doc_id] = {}
                for candidate_id, type_id in doc_dict.items():
                    results[doc_id][candidate_id] = type_id


        output_path = os.path.join(config.output_dir, 'result_' + config.test_path.split('/')[-1].split('_')[1] + '.jsonl')
        with open(output_path, "w", encoding="utf8") as writer:
            with open(config.test_path, "r") as fin:
                lines = fin.readlines()
                for line in lines:
                    doc = json.loads(line)
                    res = {}
                    doc_id = doc['id']
                    res['id'] = doc_id
                    res['predictions'] = []
                    for mention in doc['candidates']:
                        mention_id = mention['id']
                        res['predictions'].append({"id": mention_id, "type_id": results[doc_id][mention_id]})
                    writer.write(json.dumps(res, ensure_ascii=False) + "\n")
    dist.destroy_process_group()


def pseudo_label(config, model, eval_dataloader, mode):
    model.eval()
    total_loss = 0.0
    results = {}
    true = []
    pred = []
    with torch.no_grad():
        for data in tqdm(eval_dataloader, desc="Evaluating", disable=(dist.get_rank()!=0)):
            input_ids = data['input_ids']
            token_type_ids = data['token_type_ids']
            attention_mask = data['attention_mask']
            candidate_mask = data['candidate_mask']
            label_ids = data['label_ids']
            candidates = data['candidates']
            guids = data['guids']
            input_ids = torch.tensor(input_ids, dtype=torch.int64, device=config.device)            # (batch_size, pad_size)
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.int64, device=config.device)  # (batch_size, pad_size)
            attention_mask = torch.tensor(attention_mask, dtype=torch.bool, device=config.device)   # (batch_size, pad_size)
            candidate_mask = torch.tensor(candidate_mask, dtype=torch.bool, device=config.device)   # (batch_size, pad_size)
            label_ids = torch.tensor(label_ids, dtype=torch.int64, device=config.device)
            
            
            logits = model(input_ids, attention_mask, token_type_ids, candidate_mask)   # (all_candidates_nums_in_batch, lable_nums)
            probs = F.softmax(logits, dim=-1)                                           # (all_candidates_nums_in_batch, lable_nums)
            prediction = torch.argmax(probs, dim=-1).tolist()                           # (all_candidates_nums_in_batch,)
            mask_pred = torch.max(probs, dim=-1).values >= config.confid_threshold      # 小于阈值的样本均不输出
            assert len(mask_pred) == len(prediction)

            # 将prediction重新填充为(batch_size, pad_size)的形状
            pred = [[] for _ in guids]
            cnt = 0
            for i, mask in enumerate(candidate_mask):
                for m in mask:
                    if m==0:
                        pred[i].append(config.pad_label_id)
                    else:
                        if mask_pred[cnt] and prediction[cnt] in hard_label_idx:
                            pred[i].append(prediction[cnt])
                        else:
                            pred[i].append(-50)     # 过滤掉的样本预测标签置为-50

                        cnt += 1
            assert cnt==len(prediction)


            for idx in range(len(guids)):   # idx是一个batch中句子的索引
                doc_id = guids[idx].split('-')[0]
                if doc_id not in results:
                    results[doc_id] = {}

                for candidate in candidates[idx]:
                    if not candidate['reduced']:
                        assert pred[idx][candidate['offset'][0]] != config.pad_label_id
                        results[doc_id][candidate['id']] = pred[idx][candidate['offset'][0]]    # 取每个candidate第一个token的预测结果作为该candidate的结果
                    else:
                        results[doc_id][candidate['id']] = -50  # 由于max_seq_len被截断的candidate则不标注


    # 输出预测结果
    collected_results = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(collected_results, results)
    
    if dist.get_rank() == 0:
        results = {}
        for result in collected_results:
            for doc_id, doc_dict in result.items():
                if doc_id not in results:
                    results[doc_id] = {}
                for candidate_id, type_id in doc_dict.items():
                    results[doc_id][candidate_id] = type_id


        output_path = os.path.join(config.output_dir, 'result_' + config.test_path.split('/')[-1].split('_')[1] + '.jsonl')
        with open(output_path, "w", encoding="utf8") as writer:
            with open(config.test_path, "r") as fin:
                lines = fin.readlines()
                for line in lines:
                    doc = json.loads(line)
                    res = {}
                    doc_id = doc['id']
                    res['id'] = doc_id
                    res['predictions'] = []
                    for mention in doc['candidates']:
                        mention_id = mention['id']
                        res['predictions'].append({"id": mention_id, "type_id": results[doc_id][mention_id]})
                    writer.write(json.dumps(res, ensure_ascii=False) + "\n")
    dist.destroy_process_group()