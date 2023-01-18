# -*--coding:utf-8 -*--
# @Time : 2022/7/25 0025 16:36
# @Author : BAY
# @File : TFPDF未改进.py
# @Software : PyCharm
import math
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import jieba.posseg as pseg
import jieba
import datetime
from datetime import timedelta as td


def fenci_word(doc):
    stop_words = []
    stop_words_dir = r'/FunctionTrail/baidu_stopwordsV1.txt'
    for stop in open(stop_words_dir, encoding='utf-8'):
        stop_words.append(stop.replace('\n', ''))
    flags = ['a', 'an', 'ad', 'b', 'i', 'j', 'l', 'n', 'nr', 'nrt',
             'ns', 'nt', 'nw', 'nz', 's', 't', 'v', 'vn', 'PEG', 'LOC', 'ORG']
    user_dict_dir = r'/FunctionTrail/user_dictV1.txt'
    jieba.load_userdict(user_dict_dir)
    words = []
    try:
        words = " ".join([k for k, flag in pseg.lcut(doc) if k not in stop_words and flag in flags
                 and len(k) > 1])
    except:
        print("'float' object has no attribute 'decode'")
    return words


# 分词后的单词向量化
def get_transformed_dataframe(documents):
    count_vectorizer = CountVectorizer(min_df=0, max_df=1.0)
    fit_transformed_data = count_vectorizer.fit_transform(documents)
    fit_transformed_dataframe = pd.DataFrame(fit_transformed_data.A,
                                             columns=count_vectorizer.get_feature_names())
    return fit_transformed_dataframe


def get_channel_words_number():
    news_list = news_dataframe.groupby('release_source_code').count().index
    return news_list


# 去除重复的title和url，并实现title分词
def _get_news_data_with_tokenized(news_dataframe):
    news_dataframe = news_dataframe.drop_duplicates(subset=['content_check']).reset_index(drop=True)
    news_dataframe['title_tokenized'] = news_dataframe.content_check.apply(lambda x: fenci_word(x))

    return news_dataframe


def _get_word_weight_dictionary():
    print('calculate words weight')
    word_weight_dictionary = {word: _get_word_weight(word) for word in tqdm(word_list.columns)}
    word_weight_dictionary = dict(sorted(word_weight_dictionary.items(), key=lambda x: x[1], reverse=True))
    # word_weight_dictionary = sorted(word_weight_dictionary.items(), reverse=True, key=operator.itemgetter(1))
    # 结果是一个列表[('格拉西莫夫', 65.47087347356877), ('苏醒', 37.67230112023586)]
    return word_weight_dictionary


def _find_word_in_channel(word, channel):
    return word_list.iloc[(news_dataframe.release_source_code == channel).to_list()][word]


# n_jc
def _get_frequency_of_term(word, channel):
    return _find_word_in_channel(word, channel).sum()


# F_jc
def get_normalized_frequency_of_term(word, channel):
    return math.sqrt(_get_frequency_of_term(word, channel) / channel_norm_dictionary[channel])


# sqrt(F_kj)
def _get_channel_norm(channel):
    channel_word_index = news_dataframe[news_dataframe.release_source_code == channel].index
    # print(" channel_word_index ",  channel_word_index)
    channel_word_list = word_list.iloc[channel_word_index].T
    # print("channel_word_list", channel_word_list)
    return math.sqrt(channel_word_list[channel_word_list.apply(lambda x: sum(x), axis=1) > 0].T.sum().apply(
        lambda x: x ** 2).sum())


# exp(n_jc/N_c)
def get_exp(word, channel):
    return math.exp(_get_frequency_of_term(word, channel) / channel_number[channel])


# W_j
def _get_word_weight(word):
    word_weight = 0  # w_j
    for c in news_code_list:
        word_weight += (get_normalized_frequency_of_term(word, c) * get_exp(word, c))
    return word_weight


def origianl_datatype(word_list, word_weight_dictionary):
    word_list = word_list.astype(float)
    for word in word_weight_dictionary:
        print(word)
        # 遍历word_list 将计算出来的权重整到这个矩阵中
        for index, row in word_list.iterrows():
            # print("遍历word_list的每一行")
            # print("第", index, "行")
            for key in row.keys():
                if key == word[0] and row[key] == 1:
                    # print(key, word[0], "row[key]；", row[key])
                    # TODO 为啥这一步赋值，只显示整数呢？
                    # print("word[1]", word[1])
                    row[key] = word[1]
                    # print("最后结果;", key, row[key])
    new_array = word_list.values
    return new_array


# 还是不稳健，这样只有一行，不是我想要的格式果真是一行
# 我想要一个文档一行
# 存成我想要的结果了
def save_csv(_time1, word_list, word_weight_dictionary):
    # print(word_weight_dictionary)
    print(type(word_list))
    # word_list很重要
    word_list.to_csv('../corpus/统计每个文档出现的次数.csv')
    word_list = word_list.astype(float)
    # 遍历word_list 将计算出来的权重整到这个矩阵中
    for index, row in word_list.iterrows():
        # print("遍历word_list的每一行")
        print("第", index, "行")
        for key in row.keys():
            if row[key] >= 1:
                # TODO 为啥这一步赋值，只显示整数呢？
                row[key] = word_weight_dictionary[key]
                # print("最后结果;", key, row[key])
    # print("最后的word_list；\n", word_list)
    word_list.to_csv('../corpus/' + _time1 + '_result_tfpdf.csv', index=False)
    print('../corpus/' + _time1 + '_result_tfpdf.csv' + '存储完成')


if __name__ == '__main__':
    # 测试数据
    news_data = pd.read_csv(r'/corpus/test_mini_data_process.csv')
    # TODO 1、按天计算
    news_data['time'] = news_data['release_time'].apply(lambda x: str(x)[0:10])
    start_time = datetime.date(*map(int, news_data['time'].min().split('-')))
    for time_i in range(0, 1):
        print("变量循环", time_i)
        _time1 = (start_time + td(days=time_i)).strftime("%Y-%m-%d")
        print("当前的日期", _time1)
        temp_data = news_data[news_data['time'] == _time1]
        # 分词
        news_dataframe = _get_news_data_with_tokenized(temp_data)
        # print("调用'_get_news_data_with_tokenizednews_'之后的dataframe: \n", news_dataframe.head(5))
        word_list = get_transformed_dataframe(news_dataframe.title_tokenized)
        # print("单词向量化：\n", word_list)  # 1053*13283
        channel_number = news_dataframe.loc[:, ['release_source_code', 'release_source']].groupby(
            'release_source_code').count().to_dict()['release_source']
        print("按press_code排序 channel_number：\n", channel_number)
        # 还需要对数据做下处理 增加release_source_code列并与release_source相对应
        news_code_list = get_channel_words_number()
        # Float64Index([0.5, 0.7, 1.0], dtype='float64', name='release_source_code')
        print("调用'get_channel_words_number'之后的news_code_list：\n", news_code_list)
        # # # # TODO 不是很理解
        channel_norm_dictionary = {news_code: _get_channel_norm(news_code) for news_code in news_code_list}
        print(" channel_norm_dictionary：\n", channel_norm_dictionary)
        word_weight_dictionary = _get_word_weight_dictionary()
        # print(type(word_weight_dictionary))
        print(word_weight_dictionary)
        # print("word_weight_dictionary: \n", word_weight_dictionary)
        # save_csv(_time1, word_list, word_weight_dictionary)
        # X = origianl_datatype(word_list, word_weight_dictionary)
        # np.save('../corpus/' + _time1 + '_result_tfpdf.npy', X)

