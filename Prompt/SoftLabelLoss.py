import torch.nn as nn
import torch
import torch.nn.functional as F


class SoftLabelCELoss(nn.Module):
    
    def __init__(self, rlt_metric, soft_weight=0.9, epsilon=1e-9):
        """
        realitive_classes 一共拥有 num_crime（罪名个数）行， num_class （分类个数）列，
        每一行（罪名）中对应可能的类别被标记为1；
        """
        super(SoftLabelCELoss, self).__init__()
        
        self.epsilon = epsilon
        self.soft_weight = soft_weight
        self.realitive_classes = rlt_metric  # shape: [num_crime, num_class]
    
    def forward(self, inputs, target, class_id):
        """
        class_id: 该句子所在doc对应的罪名id
        """
        rm = self.realitive_classes.clone()[class_id]
        norm_realitive_classes = rm / rm.sum(-1).unsqueeze(-1).expand(rm.shape) * (1 - self.soft_weight)

        idx = target.view(-1, 1).long()
        one_hot_key = torch.zeros(idx.size(0), rm.shape[-1], device=idx.device).scatter_(1, idx, 1)

        soft_weight_metric = norm_realitive_classes + one_hot_key * self.soft_weight

        logits = F.softmax(inputs, dim=-1)
        
        loss = -soft_weight_metric * (logits + self.epsilon).log()
        loss = loss.sum(1).mean()

        return loss