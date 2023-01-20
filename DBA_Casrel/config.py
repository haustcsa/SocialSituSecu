from fastNLP import Vocabulary
from transformers import BertTokenizer, AdamW
import torch
import json
class Config:
    """
    句子最长长度是294 这里就不设参数限制长度了,每个batch 自适应长度
    """

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.bert_path = './bert-duie-chinese'

        self.num_rel = 18

        self.train_data_path = 'dataset/train.json'
        self.dev_data_path = 'dataset/dev.json'
        self.test_data_path = 'dataset/test.json'

        self.batch_size = 16

        self.rel_dict_path = 'dataset/rel.json'
        id2rel = json.load(open(self.rel_dict_path, encoding='utf8'))
        self.rel_vocab = Vocabulary(unknown=None, padding=None)
        self.rel_vocab.add_word_lst(list(id2rel.values()))  # 关系到id的映射

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = 1e-5
        self.bert_dim = 768


        self.epochs = 10
