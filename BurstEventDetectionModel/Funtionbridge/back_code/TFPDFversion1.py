# -*--coding:utf-8 -*--
# @Time : 2022/6/27 0027 15:21
# @Author : BAY
# @File : TFPDFversion2.py
# @Software : PyCharm

import math
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import jieba.posseg as pseg
import jieba
import datetime
from datetime import timedelta as td
import re


def fenci_word(doc):
    stop_words = []
    stop_words_dir = r'/FunctionTrail/baidu_stopwordsV1.txt'
    for stop in open(stop_words_dir, encoding='utf-8'):
        stop_words.append(stop.replace('\n', ''))
    flags = ['a', 'an', 'ad', 'b', 'i', 'j', 'l', 'n', 'nr', 'nrt',
             'ns', 'nt', 'nw', 'nz', 's', 't', 'v', 'vn', 'PEG', 'LOC', 'ORG']
    user_dict_dir = r'/FunctionTrail/user_dictV1.txt'
    jieba.load_userdict(user_dict_dir)
    words = ''
    try:
        if len(doc) > 1:

            words = " ".join([k for k, flag in pseg.lcut(doc) if k not in stop_words and flag in flags
                     and len(k) > 1])
    except:
        print("'float' object has no attribute 'decode'")
    # print("分词后的词语", words)
    return words


# 分词后的单词向量化
def get_transformed_dataframe(documents):
    count_vectorizer = CountVectorizer(min_df=0, max_df=1.0)
    fit_transformed_data = count_vectorizer.fit_transform(documents)
    fit_transformed_dataframe = pd.DataFrame(fit_transformed_data.A,
                                             columns=count_vectorizer.get_feature_names())
    return fit_transformed_dataframe


# 去除重复的title和url，并实现title分词
def _get_news_data_with_tokenized(news_dataframe):
    news_dataframe = news_dataframe.drop_duplicates(subset=['content_check']).reset_index(drop=True)
    # news_dataframe = news_dataframe.drop_duplicates(subset=['url']).reset_index(drop=True)
    # print("news_dataframe:\n", news_dataframe)
    news_dataframe['title_tokenized'] = news_dataframe.content_check.apply(lambda x: fenci_word(x))
    return news_dataframe


def _get_word_weight_dictionary():
    print('calculate words weight')
    word_weight_dictionary = {word: _get_word_weight(word) for word in tqdm(word_list.columns)}
    word_weight_dictionary = dict(sorted(word_weight_dictionary.items(), key=lambda x: x[1], reverse=True))
    # word_weight_dictionary = sorted(word_weight_dictionary.items(), reverse=True, key=operator.itemgetter(1))
    return word_weight_dictionary


# TODO 改进 >0 有必要的 # 统计在文档集C中出现单词j的微博次数
def _find_word_in_channel(word, channel):

    return word_list.iloc[(news_dataframe.release_source_code == channel).to_list()][word] > 0


# n_jc
def _get_frequency_of_term(word, channel):
    return _find_word_in_channel(word, channel).sum()


# F_jc
def get_normalized_frequency_of_term(word, channel):
    print(word, channel, _find_word_in_channel(word, channel).sum())
    return math.sqrt(_get_frequency_of_term(word, channel) / channel_norm_dictionary[channel])


# sqrt(F_kj)
def _get_channel_norm(channel):
    channel_word_index = news_dataframe[news_dataframe.release_source_code == channel].index
    channel_word_list = word_list.iloc[channel_word_index].T
    return math.sqrt(channel_word_list[channel_word_list.apply(lambda x: sum(x), axis=1) > 0].T.sum().apply(
        lambda x: x ** 2).sum())


# exp(n_jc/N_c)
def get_exp(word, channel):
    return math.exp(_get_frequency_of_term(word, channel) / channel_number[channel])


# W_j
def _get_word_weight(word):
    word_weight = 0  # w_j
    for c in news_dataframe.groupby('release_source_code').count().index:
        word_weight += (get_normalized_frequency_of_term(word, c) * get_exp(word, c))
        word_weight = round(word_weight, 7)
    return word_weight


if __name__ == '__main__':
    news_data = pd.read_csv(r'/corpus/test_mini_data_process.csv')
    # TODO 1、按天计算
    news_data['time'] = news_data['release_time'].apply(lambda x: str(x)[0:10])
    start_time = datetime.date(*map(int, news_data['time'].min().split('-')))
    end_time = datetime.date(*map(int, news_data['time'].max().split('-')))
    time_cha = (end_time - start_time).days
    for time_i in range(0, 1):
        # 取出当天的数据
        _time1 = (start_time + td(days=time_i)).strftime("%Y-%m-%d")
        print("当前的时间", _time1)
        temp_data = news_data[news_data['time'] == _time1]
        # 将文本进行分词
        news_dataframe = _get_news_data_with_tokenized(temp_data)
        # 将文本转换成词向量 TODO 这个格式是我想要的，在这个数据基础上进行操作
        word_list = get_transformed_dataframe(news_dataframe.title_tokenized)
        # print("word_list", word_list)
        channel_number = news_dataframe.loc[:, ['release_source_code', 'release_source']].groupby(
            'release_source_code').count().to_dict()['release_source']
        print(" channel_number:\n", channel_number)
        channel_norm_dictionary = {news_code: _get_channel_norm(news_code) for news_code in news_dataframe.groupby('release_source_code').count().index}
        print(" channel_norm_dictionary：\n", channel_norm_dictionary)
        # TODO 不是很理解
        word_weight_dictionary = _get_word_weight_dictionary()
        print("word_weight_dictionary: \n", word_weight_dictionary)




