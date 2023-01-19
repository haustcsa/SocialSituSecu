# _*__coding:utf-8 _*__
# @Time :2022/5/30 0030 21:47
# @Author :bay
# @File freqincrement高.py
# @Software : PyCharm
# TODO 词频增长率公式的实现
import pandas as pd
import json
import jieba
import jieba.posseg as pseg
import datetime
from datetime import timedelta as td
import time
from collections import Counter
from operator import itemgetter


def array_fenci_word(doc):
    doc = doc.replace(" ", "")
    stop_words = []
    stop_words_dir = r'/FunctionTrail/baidu_stopwords.txt'
    for stop in open(stop_words_dir, encoding='utf-8'):
        stop_words.append(stop.replace('\n', ''))
    flags = ['a', 'an', 'ad', 'b', 'i', 'j', 'l', 'n', 'nr', 'nrt',
             'ns', 'nt', 'nw', 'nz', 's', 't', 'v', 'vn', 'PEG', 'LOC', 'ORG']
    user_dict_dir = r'/FunctionTrail/user_dict.txt'
    jieba.load_userdict(user_dict_dir)
    words = ''
    if len(doc) > 5:
        words = [k for k, flag in pseg.lcut(doc) if k not in stop_words and flag in flags and len(k) > 1]
    # print("words；", words)
    return words


def count_words(tweets_set):
    # 词频统计那块按大小输出来就这么干
    result = []
    for doc in tweets_set:
        # print("doc", doc)
        words = array_fenci_word(doc)
        result.append(words)
    print("result", result)
    # c = Counter()
    # for i in result:
    #     for x in i:
    #         if len(x) > 1 and x != '\r\n':
    #             c[x] += 1
    # # # print("打印分词的结果")
    # result_words = {}
    # for (k, v) in c.most_common(n=None):
    #     # print(k, v)
    #     result_words[k] = v
    # print("长度;", len(result_words))
    # # print("result_words；", result_words)
    # return result_words


# 4.计算词频增长率
def get_result(freqdict, temp_result, n):
    result = {}
    for key, value in freqdict.items():
        if key not in temp_result:
            # TODO 逻辑修改 temp_result[key] = 0 还是为0吧
            temp_result[key] = 0
            # print("freqdict1[key]", type(freqdict1[key]))
        result[key] = round(temp_result[key] + (value - temp_result[key]) / n, 7)
    # print("函数调用的结果 result", result)
    #  排序 从大到小
    result_sort = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
    return result_sort


def normal(temp_result):
    compare_key1 = list(temp_result)[0]
    compare_key2 = list(temp_result)[-1]
    compare_max = temp_result[compare_key1]
    compare_min = temp_result[compare_key2]
    for key in temp_result.keys():
        temp_result[key] = round((temp_result[key] - compare_min) / (compare_max - compare_min), 7)
    return temp_result


# 将计算出来的词频增长率存储为json格式
def save_result(temp_result, n):
    file_path = '../results/word_increment/' + str(n) + '_increment.json'
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(temp_result, f, ensure_ascii=False, indent=4)
    print(file_path + "存储完成")


def increment_data():
    # 1.准备数据集（500万数据集，先使用14天或者一周的数据，论文中用了10天的数据）
    data = pd.read_csv('../../corpus/data20220911/test_total_data_process2.csv').astype(str)
    # 2.按时间计算，以天为单位
    result_freqdict = []
    data['time'] = data['release_time'].apply(lambda x: str(x)[0:10])
    start_time = datetime.date(*map(int, data['time'].min().split('-')))
    # print("开始时间", start_time)
    end_time = datetime.date(*map(int, data['time'].max().split('-')))
    # print("结束时间", end_time)
    time_cha = (end_time - start_time).days + 1
    # 0开始的6月23
    for time_i in range(11, time_cha):
        # print("变量循环", time_i)
        _time1 = (start_time + td(days=time_i)).strftime("%Y-%m-%d")
        print("当前的日期", _time1)
        temp_data = data[data['time'] == _time1]
        # 统计第Tn天的词频        # 统计第Tn天的词频
        tweets_set = temp_data['content_check'].values.tolist()
        count_words(tweets_set)
        # freqdict = count_words(tweets_set)
    #     # print("freqdict", freqdict)
    #     result_freqdict.append(freqdict)
    # # print(len(result_freqdict))
    # temp_result = get_result(result_freqdict[1], result_freqdict[0], 2)
    # result = normal(temp_result)
    # save_result(result, '2022-06-24')
    # times = ['2022-06-25', '2022-06-26', '2022-06-27', '2022-06-28', '2022-06-29',
    #          '2022-06-30', '2022-07-01', '2022-07-02', '2022-07-03', '2022-07-04',
    #          '2022-07-05', '2022-07-06', '2022-07-07', '2022-07-08', '2022-07-09',
    #          '2022-07-10', '2022-07-11', '2022-07-12', '2022-07-13', ]
    # for i, j in zip(range(2, time_cha), times):
    #     temp_result = get_result(result_freqdict[i], temp_result, i+1)
    #     result = normal(temp_result)
    #     save_result(result, j)


if __name__ == '__main__':
    start_time = time.perf_counter()
    print(start_time)
    increment_data()
    end_time = time.perf_counter()
    print(end_time)
    time_cha = end_time - start_time
    print(time_cha)



