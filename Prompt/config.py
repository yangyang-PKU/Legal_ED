import torch
from transformers import AutoTokenizer, BertTokenizer
from data_process.const import WEIGHTS
import time
import os


time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())  # 记录当前时间

class Config(object):
    """配置参数"""
    def __init__(self,):
        self.if_log_file = True
        self.overwrite_cache = False              # 是否重写缓存数据
        # self.fine_metric = True                 # 是否分别输出每一类的评测指标
        # self.output_prob = True                 # 在test集上测试时，是否输出概率
        # self.output_result = True                 # 在test集上测试时，是否输出预测标签
        # self.overwrite_cache = False              # 是否重写缓存的数据
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_gpu = torch.cuda.device_count()
        # self.fp16 = False
        # self.local_rank = -1
        
        self.do_train = True
        self.do_infer = False

        self.fold = '3'                 # train和inference的时候需要指定fold
        self.name = 'FurtherPretrain'

        self.model_type = 'bert'        # 'bert' or 'roberta' (for special token)
        self.model_name_or_path = './model/bert-base-chinese/epoch_100'
        self.data_dir = './data' 
        self.train_path = os.path.join(self.data_dir + f'/fold_{self.fold}', 'train.jsonl')
        self.valid_path = os.path.join(self.data_dir + f'/fold_{self.fold}', 'valid.jsonl')
        self.test_path = './data/test_stage2_corrected.jsonl'
        self.result_dir = os.path.join('./result', 'bert-base-chinese', self.name, 'epoch100', f'fold-{self.fold}')
        self.save_dir = os.path.join('./saved', 'bert-base-chinese', self.name, 'epoch100', f'fold-{self.fold}')
        

        self.truncated = True
        self.window_size = 25
        self.DM = False
        self.weights = [1]*109
        self.label_smooth = 0.0
        self.max_grad_norm = 0


        self.require_improvement = 12000                   # 若超过1000batch效果还没提升，则提前结束训练
        self.improvement_metric = "f1"                    # "loss" or "f1"， 用什么指标来衡量模型在dev集上有进步
        self.num_epochs = 5                               # epoch数
        self.batch_size_per_gpu = 16                       # mini-batch大小
        self.gradient_accumulation_steps = 1
        self.max_seq_len = 512                            # 每句话处理成的长度(短填长切)
        self.dropout = 0.2
        self.lr = 5e-6
        self.encoder_learning_rate = 5e-6
        self.classifier_learning_rate = 5e-5
        self.warmup = True
        self.warmup_proportion = 0.1
        self.weight_decay = 0.01                              # 正则化参数
        self.eval_step = 2000                                 # 每隔多少个step在dev集上evaluate一次
        self.not_eval_until = 0.0                             # 多少个epoch之前都不在dev集上evaluate
        

        self.seed = 100
        self.pad_label_id = -100
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name_or_path) if self.model_type == 'bert' else AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.hidden_size = 768