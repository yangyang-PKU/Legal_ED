from transformers import BertModel, BertConfig, NezhaModel, XLMRobertaModel
import torch.nn as nn
import torch
from utils import all_labels
import torch.nn.functional as F


class TokenClassificationModel(nn.Module):
    def __init__(self, config):
        super(TokenClassificationModel, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.model_name_or_path)  # Bert
        # self.bert = NezhaModel.from_pretrained(config.model_name_or_path)   # Nezha
        # self.bert = XLMRobertaModel.from_pretrained(config.model_name_or_path)
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Sequential(nn.Linear(config.hidden_size, len(all_labels)))
        

    def forward(self, input_ids, attention_mask, token_type_ids, candidate_mask, label_ids=None):
        bert_out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=False) 
        # bert_out = bert_out['hidden_states'][-1]      # (bath_size, seq_len, config.hidden_size)
        bert_out = bert_out['last_hidden_state']
        # bert_out = bert_out[0]

        mask_extented = candidate_mask.unsqueeze(-1)                     # (bath_size, seq_len, 1)
        mask_extented = mask_extented.expand(bert_out.shape)             # (bath_size, seq_len, hidden_size)
        bert_out = torch.masked_select(bert_out, mask_extented)    
        bert_out = bert_out.view(-1, mask_extented.shape[2])             # (all_candidates_nums_in_batch, class_nums)
        
        bert_out = self.dropout(bert_out)       # (all_candidates_nums_in_batch, class_nums)

        logits = self.linear(bert_out)          # (all_candidates_nums_in_batch, class_nums)
        
        if label_ids is not None:
            label_ids = label_ids[candidate_mask]           # (all_candidates_nums_in_batch, class_nums)
            class_weights = torch.FloatTensor(self.config.weights).to(self.config.device)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            # loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, label_ids)

            return loss, logits

        return logits


class DMBERT(nn.Module):
    def __init__(self, config):
        super(DMBERT, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.model_name_or_path)
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Sequential(nn.Linear(config.hidden_size * 2, len(all_labels)))

    def forward(self, input_ids, attention_mask, token_type_ids, candidate_mask, candidates, label_ids=None):
        bert_out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=False)
        bert_out = bert_out['last_hidden_state']

        pooled = None
        for idx, cand in enumerate(candidates):
            last_cut_pos = -1   
            for candidate in cand:
                if not candidate['reduced']:
                    cut_pos = candidate['offset'][0]
                    if last_cut_pos == cut_pos:     # 用于去除重复的candidate
                        continue
                    L = bert_out[idx][:cut_pos].transpose(0, 1)     # hidden_size, len(TextL)
                    R = bert_out[idx][cut_pos:].transpose(0, 1)     # hidden_size, len(TextR)
                    pooledL = F.max_pool1d(L, L.shape[1])   # hidden_size, 1
                    pooledR = F.max_pool1d(R, R.shape[1])   # hidden_size, 1
                    cur_pooled = torch.cat((pooledL, pooledR), 0).transpose(0, 1)   # 1, hidden_size * 2
                    if pooled is None:
                        pooled = cur_pooled     # 1, hidden_size * 2
                    else:
                        pooled = torch.cat((pooled, cur_pooled), 0)
                    last_cut_pos = cut_pos
        pooled = self.dropout(pooled)
        logits = self.linear(pooled)          # (all_candidates_nums_in_batch, class_nums)
        

        if label_ids is not None:
            label_ids = label_ids[candidate_mask]           # (all_candidates_nums_in_batch, class_nums)
            class_weights = torch.FloatTensor(self.config.weights).to(self.config.device)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fn(logits, label_ids)

            return loss, logits

        return logits