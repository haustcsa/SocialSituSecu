# -*--coding:utf-8 -*--
# @Time : 2022/8/23 0023 17:49
# @Author : BAY
# @File : code3NTFPDF_model.py
# @Software : PyCharm

# TODO 计算TF-PDF使用的文件
import math
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import json
import datetime
from datetime import timedelta as td
from FunctionTrail.code5jieba_fenci import fenci_word
import re


class TFPDF():
    def __init__(self, news_data: pd.DataFrame):
        self.news_data = news_data
        # print(self.news_data.columns)
        self.contents = self.news_data['content_check'].values.tolist()
        self.news_dataframe = self._get_news_data_with_tokenized(news_data)
        # 统计每个press_code出现的次数
        self.channel_number = self.news_dataframe.loc[:, ['release_source_code', 'release_source']].groupby(
            'release_source_code').count().to_dict()['release_source']
        self.news_code_list = self.news_dataframe.groupby('release_source_code').count().index
        self.word_list = self.get_transformed_dataframe(self.news_dataframe.title_tokenized)
        # print("word_list:\n", self.word_list)
        self.channel_norm_dictionary = {news_code: self._get_channel_norm(news_code) for news_code in
                                        self.news_code_list}

    # 去除重复的title和url，并实现title分词
    def _get_news_data_with_tokenized(self, news_dataframe):
        news_dataframe = news_dataframe.drop_duplicates(subset=['content_check']).reset_index(drop=True)
        # news_dataframe = news_dataframe.drop_duplicates(subset=['url']).reset_index(drop=True)
        # print("news_dataframe:\n", news_dataframe)
        news_dataframe['title_tokenized'] = news_dataframe.content_check.apply(lambda x: fenci_word(x))
        return news_dataframe

    def get_transformed_dataframe(self, documents):
        count_vectorizer = CountVectorizer(min_df=0, max_df=1.0)
        fit_transformed_data = count_vectorizer.fit_transform(documents)
        fit_transformed_dataframe = pd.DataFrame(fit_transformed_data.A,
                                                 columns=count_vectorizer.get_feature_names_out())
        return fit_transformed_dataframe

    # W_j
    def _get_word_weight(self, word):
        word_weight = 0  # w_j
        # *
        for c in self.news_code_list:
            # print(c)
            word_weight += (self.get_normalized_frequency_of_term(word, c) * self.get_exp(word, c))
            word_weight = round(word_weight, 7)
        return word_weight

    # F_jc
    def get_normalized_frequency_of_term(self, word, channel):
        return math.sqrt(self._get_frequency_of_term(word, channel) / self.channel_norm_dictionary[channel])

    # sqrt(F_kj)
    def _get_channel_norm(self, channel):
        channel_word_index = self.news_dataframe[self.news_dataframe.release_source_code == channel].index
        channel_word_list = self.word_list.iloc[channel_word_index].T
        return math.sqrt(channel_word_list[channel_word_list.apply(lambda x: sum(x), axis=1) > 0].T.sum().apply(
            lambda x: x ** 2).sum())

    # exp(n_jc/N_c)
    def get_exp(self, word, channel):
        return math.exp(self._get_frequency_of_term(word, channel) / self.channel_number[channel])

    # n_jc
    def _get_frequency_of_term(self, word, channel):
        return self._find_word_in_channel(word, channel).sum()

    # TODO 改进 >0 现在有点不知道是个啥？暂时先不改，这样改了之后，结果还是有>1的
    def _find_word_in_channel(self, word, channel):
        # print("*********")
        # print(word_list.iloc[(news_dataframe.release_source_code == channel).to_list()][word])
        # return word_list.iloc[(news_dataframe.release_source_code == channel).to_list()][word] > 0
        return self.word_list.iloc[(self.news_dataframe.release_source_code == channel).to_list()][word]

    def _get_word_weight_dictionary(self):
        print('calculate words weight')
        word_weight_dictionary = {word: self._get_word_weight(word) for word in tqdm(self.word_list.columns)}
        word_weight_dictionary = dict(sorted(word_weight_dictionary.items(), key=lambda x: x[1], reverse=True))
        # word_weight_dictionary = sorted(word_weight_dictionary.items(), reverse=True, key=operator.itemgetter(1))
        return word_weight_dictionary




