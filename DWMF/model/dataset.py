import os.path
from PIL import Image
import jieba
import numpy as np
from torchvision import transforms as transforms
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from typing import Optional
from textrank4zh import TextRank4Sentence


def get_stop_words(stop_words):
    w = []
    with open(stop_words, 'r', encoding='utf-8') as f:
        for c in f.readlines():
            c = c.strip()
            w.append(c)
    return w


def get_chinese(uchar):
    if u'\u9fa5' >= uchar >= u'\u4e00':
        return True
    else:
        return False


class ClassDataset(Dataset):
    def __init__(self,
                 dataset: Optional[str],
                 img_file: Optional[str],
                 pad: Optional[int] = 254,
                 state: Optional[str] = 'train',
                 vocab_vector: Optional[str] = r'data/vocabulary_vector.csv',
                 stop_words: Optional[str] = r'data/stopwords.txt',
                 bert_config: Optional[str] = r'bert-base-chinese',
                 ):
        super(ClassDataset, self).__init__()

        self.state = state
        print('load bert tokenizer ....')
        self.tokenizer = BertTokenizer.from_pretrained(bert_config)
        print('load done ....')
        self.netdata = pd.read_csv(dataset, encoding='utf-8', usecols=['text', 'piclist', 'label'])
        self.netdata = np.array(self.netdata, dtype=np.ndarray)
        self.lstm_vector = pd.read_csv(vocab_vector, encoding='utf-8')
        self.stop_words = stop_words
        self.pad = pad
        self.img_file = img_file

        self.train = []

        for train in self.netdata:
            self.train.append(train)

        if self.state == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.RandomCrop(size=128),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

        if self.state == 'test':
            self.transforms = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

    def __getitem__(self, index):

        train_all = self.train[index]

        input_ids = self.bert_pad(train_all[0])
        bert_ids = torch.tensor([input_ids])

        bert_mask = self.mask(bert_ids)
        bert_ids = bert_ids.squeeze(0)
        bert_mask = bert_mask.squeeze(0)

        key_sentence = self.key_phrase(train_all[0])
        lstm_input = self.lstm_pad(key_sentence[0]['sentence'])

        img_name = str(train_all[1].split('.')[0]) + '.jpg'
        img_name = os.path.join(self.img_file, img_name)

        if not os.path.exists(img_name):
            img_fake = torch.randn(3, 128, 128)
            img_fake = torch.autograd.Variable(img_fake, requires_grad=False)
            img_data = img_fake.data
            label = torch.tensor(int(0), dtype=torch.float32)
        else:
            img_data = Image.open(img_name)
            img_data = img_data.convert('RGB')
            img_data = self.transforms(img_data)
            label = torch.tensor(int(train_all[2]), dtype=torch.float32)

        return bert_ids, bert_mask, img_data, lstm_input, label

    def __len__(self):
        return len(self.train)

    def mask(self, input_ids):
        input_ids = input_ids.clone().detach()
        masks = []
        for seq in input_ids:
            seq_mask = [float(i > 0) for i in seq]
            masks.append(seq_mask)
        return torch.tensor(masks)

    def bert_pad(self, bert_token):

        input_id = self.tokenizer.encode(bert_token[:self.pad])  # 直接截断
        if len(input_id) < self.pad + 2:  # 补齐（pad的索引号就是0）
            input_id.extend([0] * (self.pad + 2 - len(input_id)))
        return input_id

    def lstm_pad(self, lstm_token):

        x = jieba.cut(lstm_token, cut_all=False)
        jieba_list = ','.join(x)
        jieba_list = jieba_list.split(',')
        seg_remove_stopword = [s for s in jieba_list if s not in get_stop_words(self.stop_words)]
        seg_remove_symbol = [s for s in seg_remove_stopword if get_chinese(s) is True]
        pad = ['PAD'] * 64
        seg_remove_symbol = seg_remove_symbol + pad
        seg_remove_symbol = seg_remove_symbol[:64]

        key_ = []
        for i, key in enumerate(seg_remove_symbol):
            if key in self.lstm_vector.keys():
                key = self.lstm_vector[key]
            else:
                key = np.zeros(shape=(100,))
            key_.append(key)
        key_ = np.array(key_)
        return torch.tensor(key_, dtype=torch.float32)

    def key_phrase(self, text):
        tr4s = TextRank4Sentence()
        tr4s.analyze(text, lower=True, source='all_filters')
        # text    -- 文本内容，字符串
        # lower   -- 是否将英文文本转换为小写，默认值为False
        # source  -- 选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来生成句子之间的相似度。
        # 		  -- 默认值为`'all_filters'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'
        # sim_func -- 指定计算句子相似度的函数
        # 获取最重要的num个长度大于等于sentence_min_len的句子用来生成摘要
        keysentences = tr4s.get_key_sentences(num=1, sentence_min_len=1)
        return keysentences


if __name__ == '__main__':

    img_file_ = r'train_img'
    dataSwt = ClassDataset(dataset=r'data/train.csv',
                           img_file=img_file_,
                           state='train')
    dataloader = DataLoader(dataSwt, batch_size=10, shuffle=True)
    for i, batch in enumerate(dataloader):
        pass
        # bert_ids, bert_mask, lstm_input, img_data, label
        print('bert_ids ------------- >:', batch[0].shape)
        print('bert_mask ------------- >:', batch[1].shape)
        print('img_data ------------- >:', batch[2].shape)
        print('lstm_data ------------- >:', batch[3].shape)
        print('label ------------- >:', batch[4])
