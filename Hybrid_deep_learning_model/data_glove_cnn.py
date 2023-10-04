import pandas as pd
import torch
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import gc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_auc_score
import torchtext.vocab as vocab
from tqdm import tqdm
import os
from torchtext import data

# 将模型和输入数据都设置为 GPU 类型
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

def data_deal():
    #------------------以下代码不包含文本处理模型------------------#
    # 读取用户数据集
    users_df = pd.read_csv('users.csv', encoding='utf-8', low_memory=False)
    # 选取部分特征
    users_df = users_df[["id", "statuses_count", "followers_count", "friends_count", "favourites_count", "listed_count", "description", "test_set_2"]]
    # users_df['id'] = users_df['id'].astype("int64")
    users_df = users_df.rename(columns={'id': 'user_id'})

    # 读取推文数据集
    tweets_df = pd.read_csv('tweets.csv', encoding='utf-8', low_memory=False)
    tweets_df = tweets_df[["id", "user_id", "source", "text", "retweet_count", "reply_count", "favorite_count"]]
    tweets_df = tweets_df.rename(columns={'id': 'tweet_id'})
    #删除最后一行
    tweets_df = tweets_df.drop(tweets_df.index[-1])

    #合并两个数据集，组成元数据
    metadata_df = pd.merge(users_df, tweets_df, on='user_id')

    # 删除包含缺失值的行
    metadata_df.dropna(subset=['source', 'text'], inplace=True)

    # 使用前一个非缺失值填充 description
    metadata_df['description'].fillna(method='ffill', inplace=True)

    grouped_metadata_df = metadata_df.groupby('user_id')
    #控制每个用户的推文数
    tweets_num = 20    #每个用户的推文数 #-----------------------------------------------
    user_num = tweets_num * 30   #一共多少条为0的数据  #一共多少条为1的数据   #-----------------------------------------------
    metadata_row = grouped_metadata_df.head(tweets_num)
    metadata_df = pd.DataFrame(metadata_row)
    # print(metadata_df)
    
    # 选取test_set_2为0的前20个users_df
    #若每个用户推文数量为2且要保证每个用户有4个为0的数据，head要设置为8
    test_set_2_0 = metadata_df[metadata_df['test_set_2'] == 0].head(user_num)   #一共多少条为0的数据  #-----------------------------------------------
    #最多490个为1的数据
    test_set_2_1 = metadata_df[metadata_df['test_set_2'] == 1].head(user_num)   #一共多少条为1的数据  #-----------------------------------------------
    # print(test_set_2_1)

    # 将两个DataFrame合并成一个
    new_users_df = pd.concat([test_set_2_0, test_set_2_1], ignore_index=True)
    # 打乱顺序
    metadata_df = new_users_df.sample(frac=1).reset_index(drop=True)
    # print(metadata_df)
    # print(metadata_df['test_set_2'])

    # #输出表头
    # # print(metadata_df.columns.values.tolist())

    #判断是否有缺失值
    # print(metadata_df.isnull().any())
    # #------------------以上代码不包含文本处理模型------------------#

    #用户
    #id、用户发表的推文总数、关注此用户的其他用户数、此用户关注的其他用户数、此用户已喜欢的推文数、此用户出现在的媒体列表和其他用户列表中的次数、个人描述
    # user_id、statuses_count、followers_count、friends_count、favourites_count、listed_count、description
    #推文
    #tweet_id、推文被转发的次数、推文被回复的次数、推文被赞的次数、推文的来源（例如，Twitter网站、移动应用程序等）、推文文本的内容
    # tweet_id、retweet_count、reply_count、favorite_count、source、text

    #------------------以下代码包含文本处理模型------------------#
    # 选取元数据中的数字部分
    # numeric_columns = ['user_id', 'statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count', 'tweet_id', 'retweet_count', 'reply_count', 'favorite_count']
    #tweet元数据
    tweet_numeric_columns = ['user_id', 'tweet_id', 'retweet_count', 'reply_count', 'favorite_count']
    # 将 metadata_df 中的数字部分转换为 Tensor 类型
    metadata_df['tweet_id'] = metadata_df['tweet_id'].astype('float32')
    metadata_df['retweet_count'] = metadata_df['retweet_count'].astype('float32')
    metadata_df['reply_count'] = metadata_df['reply_count'].astype('float32')
    metadata_df['favorite_count'] = metadata_df['favorite_count'].astype('float32')

    #判断是否有非数值型数据
    # print(metadata_df[numeric_columns].dtypes)

    glove_path = 'glove.twitter.27B.200d.txt'

    #使用预训练词向量将文本转换为数字向量
    # 获取预训练的词向量（这里以 GloVe 为例）
    vectors = vocab.Vectors(name=glove_path)
    # print(vectors)
    # 获取文本字段的词汇表
    tokenizer = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
    # print(TEXT)
    TEXT.build_vocab(metadata_df['text'], vectors=vectors)

    #处理metadata_df中的 text 文本数据
    tweet_text_transformed_tensor = torch.tensor(TEXT.process(metadata_df['text']).numpy()).to(device)

    #处理metadata_df中的 description 文本数据
    DESCRIPTION = data.Field(sequential=True, tokenize=tokenizer, lower=True)
    DESCRIPTION.build_vocab(metadata_df['description'], vectors=vectors)
    description_transformed_tensor = torch.tensor(DESCRIPTION.process(metadata_df['description']).numpy()).to(device)

    #处理metadata_df中的 source 文本数据
    SOURCE = data.Field(sequential=True, tokenize=tokenizer, lower=True)
    SOURCE.build_vocab(metadata_df['source'], vectors=vectors)
    source_transformed_tensor = torch.tensor(SOURCE.process(metadata_df['source']).numpy()).to(device)
    # 把[84,6]转换为[6,84]
    source_transformed_tensor = source_transformed_tensor.transpose(0, 1)
    # 将 metadata_df 中的数字部分转换为 Tensor 类型
    tweet_metadata_numeric_tensor = torch.tensor(metadata_df[tweet_numeric_columns].values).to(device)


    tweet_metadata_combined_tensor = torch.cat([tweet_metadata_numeric_tensor, source_transformed_tensor], dim=1)
    #调换tweet_metadata_combined_tensor的列顺序
    tweet_text_transformed_tensor = tweet_text_transformed_tensor.transpose(0, 1)
    description_transformed_tensor = description_transformed_tensor.transpose(0, 1)
    print(tweet_metadata_combined_tensor.shape)
    print(tweet_text_transformed_tensor.shape)
    print(description_transformed_tensor.shape)
    
    return tweet_metadata_combined_tensor, tweet_text_transformed_tensor, description_transformed_tensor, metadata_df

# data_deal()
class TweetLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout=0.0002):
        super(TweetLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout


    def forward(self, x):
        embedded = self.embedding(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        # out, _ = self.lstm(embedded, (h0, c0))
        out, _ = self.lstm(F.dropout(embedded, p=self.dropout, training=self.training), (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class MetadataLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(MetadataLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        x = x.unsqueeze(1)  # 将输入从[batch_size, input_dim]转换为[batch_size, seq_len, input_dim]
        #转换数据类型
        h0 = h0.float()
        c0 = c0.float()
        # print(x.shape, x.dtype)
        # print(h0.shape, h0.dtype)
        # print(c0.shape, c0.dtype)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class AccountDescCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, kernel_sizes, out_channels):
        super(AccountDescCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        #if not isinstance(kernel_sizes, (tuple, list)):
        # print(type(kernel_sizes))
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, out_channels, kernel_size=k) for k in kernel_sizes])
        self.fc = nn.Linear(out_channels * len(kernel_sizes), num_classes)

    def forward(self, x):
        embedded = self.embedding(x).transpose(1, 2)
        x = [F.relu(conv(embedded)) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.fc(x)
        return x


class SocialBotDetector(nn.Module):
    def __init__(self, tweet_vocab_size, tweet_embedding_dim, tweet_hidden_dim, tweet_num_layers, metadata_input_dim,
                 metadata_hidden_dim, metadata_num_layers, desc_vocab_size, desc_embedding_dim, desc_out_channels,
                 kernel_sizes, num_classes):
        super(SocialBotDetector, self).__init__()

        self.tweet_lstm = TweetLSTM(tweet_vocab_size, tweet_embedding_dim, tweet_hidden_dim, tweet_num_layers, num_classes)
        self.metadata_lstm = MetadataLSTM(metadata_input_dim, metadata_hidden_dim, metadata_num_layers, num_classes)
        #kernel_sizes转为list类型
        if not isinstance(kernel_sizes, (list, tuple)):  # 如果不是列表或元组类型
            kernel_sizes = [kernel_sizes]  # 转换为只包含一个元素的列表类型
        # print(type(kernel_sizes))
        self.desc_cnn = AccountDescCNN(desc_vocab_size, desc_embedding_dim, num_classes, list(kernel_sizes), desc_out_channels)
        self.fc = nn.Linear(num_classes * 3, num_classes)

    def forward(self, tweet_x, metadata_x, desc_x):
        tweet_out = self.tweet_lstm(tweet_x)
        metadata_out = self.metadata_lstm(metadata_x)
        desc_out = self.desc_cnn(desc_x)
        combined_out = torch.cat((tweet_out, metadata_out, desc_out), dim=1)
        out = self.fc(combined_out)
        return out

def train_and_eval():
    # 加载数据集
    print("Loading data...")
    tweet_metadata_combined_tensor, tweet_text_transformed_tensor, description_transformed_tensor, metadata_df = data_deal()
    X_tweet_train, X_tweet_test, y_train, y_test = train_test_split(tweet_text_transformed_tensor.cpu().numpy(), metadata_df["test_set_2"].values, test_size=0.5, random_state=42)
    X_metadata_train, X_metadata_test, _, _ = train_test_split(tweet_metadata_combined_tensor.cpu().numpy(), metadata_df['test_set_2'].values, test_size=0.5, random_state=42)
    X_desc_train, X_desc_test, _, _ = train_test_split(description_transformed_tensor.cpu().numpy(), metadata_df['test_set_2'].values, test_size=0.5, random_state=42)

    # metadata_df中有多少个数据
    # 当前模型的参数，可根据需要进行修改
    tweet_vocab_size = 100000
    tweet_embedding_dim = 256
    tweet_hidden_dim = 128
    tweet_num_layers = 2
    metadata_input_dim = X_metadata_train.shape[1]
    metadata_hidden_dim = 64
    metadata_num_layers = 1
    desc_vocab_size = 100000
    desc_embedding_dim = 256
    desc_hidden_dim = 64
    desc_num_layers = 1
    num_classes = 1
    #学习率
    lr = 0.001
    num_epochs = 2
    batch_size = 1
    #L2正则化参数
    weight_decay = 0.001

    # 构建 SocialBotDetector 模型
    social_bot_detector = SocialBotDetector(tweet_vocab_size, tweet_embedding_dim, tweet_hidden_dim, tweet_num_layers,
                                             metadata_input_dim, metadata_hidden_dim, metadata_num_layers,
                                             desc_vocab_size, desc_embedding_dim, desc_hidden_dim, desc_num_layers,
                                             num_classes).to(device)

    # # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.RMSprop(social_bot_detector.parameters(), lr=lr, weight_decay=weight_decay, alpha=alpha)
    optimizer = optim.Adam(social_bot_detector.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = optim.SGD(social_bot_detector.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    # torch.cuda.empty_cache()

    # # 训练和评估模型
    for epoch in range(num_epochs):
        social_bot_detector.train()
        total_loss = 0
        num_correct = 0
        num_total = 0
        # for i in range(0, len(X_tweet_train), batch_size):
        for i in tqdm(range(0, len(X_tweet_train), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}"):
            end_idx = i + batch_size if i + batch_size < len(X_tweet_train) else len(X_tweet_train)
            tweet_x = torch.tensor(X_tweet_train[i:end_idx]).to(device)
            metadata_x = torch.tensor(X_metadata_train[i:end_idx]).to(device)
            desc_x = torch.tensor(X_desc_train[i:end_idx]).to(device)
            label = torch.tensor(y_train[i:end_idx], dtype=torch.float32).unsqueeze(1).to(device)

            optimizer.zero_grad()
            # print(tweet_x.shape, metadata_x.shape, desc_x.shape)
            #转换数据类型
            tweet_x = tweet_x.long()
            metadata_x = metadata_x.float()
            desc_x = desc_x.long()
            pred = social_bot_detector(tweet_x, metadata_x, desc_x)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * (end_idx - i)
            num_correct += ((pred > 0) == (label > 0)).sum().item()
            num_total += (end_idx - i)

        train_loss = total_loss / num_total
        train_acc = num_correct / num_total
        

        social_bot_detector.eval()
        with torch.no_grad():
            total_loss = 0
            num_correct = 0
            num_total = 0
            # for i in range(0, len(X_tweet_test), batch_size):
            for i in tqdm(range(0, len(X_tweet_test), batch_size), desc="Testing"):
                end_idx = i + batch_size if i + batch_size < len(X_tweet_test) else len(X_tweet_test)
                tweet_x = torch.tensor(X_tweet_test[i:end_idx]).to(device)
                metadata_x = torch.tensor(X_metadata_test[i:end_idx]).to(device)
                desc_x = torch.tensor(X_desc_test[i:end_idx]).to(device)
                label = torch.tensor(y_test[i:end_idx], dtype=torch.float32).unsqueeze(1).to(device)
                #转换数据类型
                tweet_x = tweet_x.long()
                metadata_x = metadata_x.float()
                desc_x = desc_x.long()

                pred = social_bot_detector(tweet_x, metadata_x, desc_x)
                loss = criterion(pred, label)

                total_loss += loss.item() * (end_idx - i)
                num_correct += ((pred > 0) == (label > 0)).sum().item()
                num_total += (end_idx - i)

            val_loss = total_loss / num_total
            val_acc = num_correct / num_total

        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'
              .format(epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc))
    
    # 计算模型的评估指标
    with torch.no_grad():
        social_bot_detector.eval()
        tweet_x = torch.tensor(X_tweet_test).to(device)
        metadata_x = torch.tensor(X_metadata_test).to(device)
        desc_x = torch.tensor(X_desc_test).to(device)
        label = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)
        #转换数据类型
        tweet_x = tweet_x.long()
        metadata_x = metadata_x.float()
        desc_x = desc_x.long()


        pred = social_bot_detector(tweet_x, metadata_x, desc_x)
        pred_label = (pred > 0).int().squeeze().cpu().numpy()
        # print(y_test)
        # print(pred_label)

        accuracy = accuracy_score(y_test, pred_label)
        precision = precision_score(y_test, pred_label)
        recall = recall_score(y_test, pred_label)
        f1 = f1_score(y_test, pred_label)

        # 计算混淆矩阵
        cm = confusion_matrix(y_test, pred_label)
        # 计算 TN, FP, TP, FN
        tn, fp, fn, tp = cm.ravel()

        # 计算 specificity 和 MCC
        specificity = tn / (tn + fp)
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        print('Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}, specificity: {:.4f}, mcc: {:.4f}'.format(accuracy, precision, recall, f1, specificity, mcc))

        with torch.no_grad():
            social_bot_detector.eval()
            tweet_x = torch.tensor(X_tweet_train).to(device)
            metadata_x = torch.tensor(X_metadata_train).to(device)
            desc_x = torch.tensor(X_desc_train).to(device)
            label = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
            #转换数据类型
            tweet_x = tweet_x.long()
            metadata_x = metadata_x.float()
            desc_x = desc_x.long()

            pred_train = social_bot_detector(tweet_x, metadata_x, desc_x).cpu().numpy()
            auc_train = roc_auc_score(y_train, pred_train)

            tweet_x = torch.tensor(X_tweet_test).to(device)
            metadata_x = torch.tensor(X_metadata_test).to(device)
            desc_x = torch.tensor(X_desc_test).to(device)
            label = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)
            #转换数据类型
            tweet_x = tweet_x.long()
            metadata_x = metadata_x.float()
            desc_x = desc_x.long()

            pred_test = social_bot_detector(tweet_x, metadata_x, desc_x).cpu().numpy()
            auc_test = roc_auc_score(y_test, pred_test)

        # # 如果当前 AUC 更好，保存模型
        # if auc_test > best_auc:
        #     torch.save(social_bot_detector.state_dict(), 'best_model.pth')
        #     best_auc = auc_test
        
        # 输出 AUC 值
        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}, Train AUC: {:.4f}, Val AUC: {:.4f}'
                .format(epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc, auc_train, auc_test))

train_and_eval()