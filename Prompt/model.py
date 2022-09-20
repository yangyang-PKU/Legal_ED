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
        bert_out = bert_out['last_hidden_state']

        mask_extented = candidate_mask.unsqueeze(-1)                     # (bath_size, seq_len, 1)
        mask_extented = mask_extented.expand(bert_out.shape)             # (bath_size, seq_len, hidden_size)
        bert_out = torch.masked_select(bert_out, mask_extented)    
        bert_out = bert_out.view(-1, mask_extented.shape[2])             # (all_candidates_nums_in_batch, class_nums)
        
        bert_out = self.dropout(bert_out)       # (all_candidates_nums_in_batch, class_nums)

        logits = self.linear(bert_out)          # (all_candidates_nums_in_batch, class_nums)
        
        if label_ids is not None:
            label_ids = label_ids[candidate_mask]           # (all_candidates_nums_in_batch, class_nums)
            class_weights = torch.FloatTensor(self.config.weights).to(self.config.device)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=self.config.label_smooth)
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

    def forward(self, input_ids, attention_mask, token_type_ids, candidate_mask, maskL, maskR, truncated, label_ids=None):
        bert_out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=False)
        bert_out = bert_out['last_hidden_state']
        batch_size = input_ids.size(0)

        conv = bert_out.permute(2, 0, 1)  # bs , seq , hidden  ->  hidden , bs , seq
        L = (conv * maskL).transpose(0, 1)  # maskL.shape: bs , seq   - >  bs , hidden , seq
        R = (conv * maskR).transpose(0, 1)

        L = L + torch.ones_like(L)      # add one to avoid overflowing
        R = R + torch.ones_like(R)      # bs , hidden , seq

        pooledL = F.max_pool1d(L, L.shape[-1]).contiguous().view(batch_size, self.config.hidden_size)        # bs , hidden
        pooledR = F.max_pool1d(R, R.shape[-1]).contiguous().view(batch_size, self.config.hidden_size)        # bs , hidden
        pooled = torch.cat((pooledL, pooledR), 1)   # bs , 2*hidden
        pooled = pooled - torch.ones_like(pooled)
        pooled = self.dropout(pooled)
        logits = self.linear(pooled)        # bs , class_nums
        truncated = (truncated==0).unsqueeze(-1)                     # (bath_size, 1)
        truncated = truncated.expand(logits.shape)             # (bath_size, class_nums)
        logits = torch.masked_select(logits, truncated).view(-1, logits.shape[-1])      # (all_candidates_nums_in_batch, class_nums)
        

        if label_ids is not None:
            label_ids = label_ids[candidate_mask]           # (all_candidates_nums_in_batch, class_nums)
            class_weights = torch.FloatTensor(self.config.weights).to(self.config.device)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fn(logits, label_ids)

            return loss, logits

        return logits