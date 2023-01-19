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
import time
# 4.2918329


def fenci_word(doc):
    stop_words = []
    stop_words_dir = r'../../FunctionTrail/baidu_stopwords.txt'
    for stop in open(stop_words_dir, encoding='utf-8'):
        stop_words.append(stop.replace('\n', ''))
    flags = ['a', 'an', 'ad', 'b', 'i', 'j', 'l', 'n', 'nr', 'nrt',
             'ns', 'nt', 'nw', 'nz', 's', 't', 'v', 'vn', 'PEG', 'LOC', 'ORG']
    user_dict_dir = r'../../FunctionTrail/user_dict.txt'
    jieba.load_userdict(user_dict_dir)
    words = []
    try:
        words = " ".join([k for k, flag in pseg.lcut(doc) if k not in stop_words and flag in flags
                 and len(k) > 1])
    except:
        print("'float' object has no attribute 'decode'")
    return words


# 去除重复的title和url，并实现title分词
def _get_news_data_with_tokenized(news_dataframe):
    news_dataframe = news_dataframe.drop_duplicates(subset=['content_check']).reset_index(drop=True)
    news_dataframe['title_tokenized'] = news_dataframe.content_check.apply(lambda x: fenci_word(x))

    return news_dataframe


if __name__ == '__main__':
    start_time = time.perf_counter()
    print(start_time)
    # 测试数据
    news_data = pd.read_csv(r'../../corpus/test_mini_data_process.csv')
    # TODO 1、按天计算
    news_data['time'] = news_data['release_time'].apply(lambda x: str(x)[0:10])
    start_time = datetime.date(*map(int, news_data['time'].min().split('-')))
    for time_i in range(0, 5):
        print("变量循环", time_i)
        _time1 = (start_time + td(days=time_i)).strftime("%Y-%m-%d")
        print("当前的日期", _time1)
        temp_data = news_data[news_data['time'] == _time1]
        # 分词
        news_dataframe = _get_news_data_with_tokenized(temp_data)
        print("调用'_get_news_data_with_tokenizednews_'之后的dataframe: \n", news_dataframe.head(5))
        # word_list = get_transformed_dataframe(news_dataframe.title_tokenized)
        # # print("单词向量化：\n", word_list)  # 1053*13283
    end_time = time.perf_counter()
    print(end_time)
    time_cha = end_time - start_time
    print(time_cha)