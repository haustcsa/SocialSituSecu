# coding: UTF-8
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained_bert import BertModel, BertTokenizer


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'  # 训练集
        self.dev_path = dataset + '/data/dev.txt'  # 验证集
        self.test_path = dataset + '/data/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]  # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 15  # epoch数
        self.batch_size = 8  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 852


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv6 = nn.Conv2d(16, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x1 = self.conv2(x)
        x1 = self.conv3(x1)
        x = x + x1
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = self.bn(x)
        return x


class Model(nn.Module):

    def __init__(self, config, pic_network):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.pic_network = pic_network

        self.pic_spatial_attention_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.pic_spatial_attention_conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.pic_spatial_attention_conv3 = nn.Conv2d(16, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

        self.pic_channel_attention_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.pic_channel_attention_conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.pic_channel_attention_conv3 = nn.Conv2d(16, 16, 13)

        self.relu = nn.ReLU()
        self.pic_mapping1 = nn.Linear(2704, 512)
        self.pic_mapping2 = nn.Linear(512, 512)
        
        self.fusion_attention1 = nn.Linear(1280, 128)
        self.fusion_attention2 = nn.Linear(128, 2)

        self.fc1 = nn.Linear(1280, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, config.num_classes)
        
        self.dropout = nn.Dropout(p=0.1)

        self.softmax = nn.Softmax()


    def forward(self, x, x_img):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        img_out = self.pic_network(x_img)
        
        img_out_spatial_attention = self.pic_spatial_attention_conv1(img_out)
        img_out_spatial_attention = self.pic_spatial_attention_conv2(img_out_spatial_attention)
        img_out_spatial_attention = img_out + img_out_spatial_attention
        img_out_spatial_attention = self.pic_spatial_attention_conv3(img_out_spatial_attention)
        img_out_spatial_attention = self.sigmoid(img_out_spatial_attention)
        # print('spatial', img_out_spatial_attention.shape)
        img_out = img_out_spatial_attention * img_out
        
        img_out_channel_attention = self.pic_channel_attention_conv1(img_out)
        img_out_channel_attention = self.pic_channel_attention_conv2(img_out_channel_attention)
        img_out_channel_attention = img_out + img_out_channel_attention
        img_out_channel_attention = self.pic_channel_attention_conv3(img_out_channel_attention)
        img_out_channel_attention = self.sigmoid(img_out_channel_attention)
        # print('channel', img_out_channel_attention.shape)
        img_out = img_out_channel_attention * img_out

        img_out = img_out.flatten(1)
        # print('img_out:', img_out.shape)

        infor_pic = self.relu(self.pic_mapping1(img_out))
        infor_pic = self.pic_mapping2(infor_pic)
        # print('infor_pic:', infor_pic.shape)

        fusion_feature = torch.cat([pooled, infor_pic], 1)
        # print('fusion_shape:', fusion_feature.shape)
        fusion_feature_attention = self.fusion_attention1(fusion_feature)
        fusion_feature_attention = self.fusion_attention2(fusion_feature_attention)
        fusion_feature_attention = self.sigmoid(fusion_feature_attention)
        # print(fusion_feature_attention[:, 0].unsqueeze(0).shape)
        pooled = pooled * fusion_feature_attention[:, 0].unsqueeze(1)
        infor_pic = infor_pic * fusion_feature_attention[:, 1].unsqueeze(1)

        final = torch.cat([pooled, infor_pic], 1)
        out = self.relu(self.fc1(final))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        out = self.dropout(out)
        out = self.softmax(out)
        return out
