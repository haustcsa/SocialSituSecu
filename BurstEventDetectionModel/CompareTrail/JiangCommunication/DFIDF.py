# _*__coding:utf-8 _*__
# @Time :2022/5/10 0010 16:32
# @Author :bay
# @File DFIDF.py
# @Software : PyCharm
# TODO 实现计算DF-IDF
import math
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from FunctionTrail.code5jieba_fenci import fenci_word
from datetime import date, timedelta as td
import datetime
import json

file_path = r'../CompareTrail/JiangCommunication/word_dict_ttV2.json'
# print(file_path)
f = open(file_path, 'r', encoding='utf-8')
word_dic_tt = json.load(f)


def get_df():
    word_dics = {}
    times = ['2022-06-24', '2022-06-25', '2022-06-26', '2022-06-27', '2022-06-28',
             '2022-06-29', '2022-06-30', '2022-07-01', '2022-07-02', '2022-07-03',
             '2022-07-04', '2022-07-05', '2022-07-06', '2022-07-07', '2022-07-08',
             '2022-07-09', '2022-07-10', '2022-07-11', '2022-07-12', '2022-07-13']
    for i in range(0, 20):
        print("times[i]", times[i])
        word_dics[i] = word_dic_tt[times[i]]
    return word_dics


# 第t-τ~t天内单词j的平均DF，一般τ=14
def get_idf():
    # 时间循环
    t = 1
    new_dicts = {}
    word_dics = {}
    times = ['2022-06-23', '2022-06-24', '2022-06-25', '2022-06-26', '2022-06-27', '2022-06-28',
             '2022-06-29', '2022-06-30', '2022-07-01', '2022-07-02', '2022-07-03',
             '2022-07-04', '2022-07-05', '2022-07-06', '2022-07-07', '2022-07-08',
             '2022-07-09', '2022-07-10', '2022-07-11', '2022-07-12', '2022-07-13']
    n = 2
    m = 0
    for i in range(1, 21):
        print("time1", times[i])
        # print('\n')
        new_dict = {}
        for j in range(m, n):
            print("time2", times[j])
            word_dics[j] = word_dic_tt[times[j]]
        print('**************************')
        m += 1
        n += 1
        print("word_dics:", type(word_dics))
        # key_t是干啥用的？
        key_t = word_dics[1]
        # print("key_t", key_t)
        for k1 in key_t.items():
            # k1是一个元组 ('知名', 0.0027214804454652017)
            new_dict[k1[0]] = 0
            # 一个值为0的字典
            for k2, values in word_dic_tt.items():
                # k2是键，values是值也是一个字典
                for k3 in values.items():
                    #  k3是一个元组 ('知名', 0.0027214804454652017)
                    if k1[0] == k3[0]:
                        new_dict[k1[0]] += k3[1]
        new_dicts[i] = new_dict
    # print("new_dicts", new_dicts)
    # log(1+1/(x/t))
    for k4, v4 in new_dicts.items():
        for k5, v5 in v4.items():
            v5 = math.log(1+v5/t)
            v4[k5] = v5
    # print("计算过后的new_dicts", new_dicts)
    return new_dicts


def get_df_idf(word_dics , new_dics):
    for (k1, v1), (k2, v2) in zip(word_dics.items(), new_dics.items()):
        for (k3, v3), (k4, v4) in zip(v1.items(), v2.items()):
            v3 = v3 * v4
            v1[k3] = v3
    return word_dics

#
# def normal(data):
#     average = float(sum(data)) / len(data)
#     data_1 = [round(abs(x - average) / (max(data) - min(data)), 7) for x in data]
#     return data_1


# 归一化
def normal(temp_result):
    compare_key1 = list(temp_result)[0]
    compare_key2 = list(temp_result)[-1]
    compare_max = temp_result[compare_key1]
    compare_min = temp_result[compare_key2]
    for key in temp_result.keys():
        temp_result[key] = round((temp_result[key] - compare_min) / (compare_max - compare_min), 7)
    return temp_result


# if __name__ == '__main__':
def get_dfidf_value(start_time):
    word_dics = get_df()
    new_dics = get_idf()
    new_dictionary = get_df_idf(word_dics, new_dics)
    for key, key_value in new_dictionary.items():
        key_time = (start_time + td(days=key+1)).strftime("%Y-%m-%d")
        # print("key", key)
        # 排序
        key_value = dict(sorted(key_value.items(), key=lambda x: x[1], reverse=True))
        # print("key_value", key_value)
        # 将字典数据存储为json格式
        temp_value = normal(key_value)
        with open(r'../CompareTrail/JiangCommunication/word_dfidf/{0}_dtidf.json'.format(key_time), 'w',
                  encoding='utf-8') as f:
            json.dump(temp_value, f, ensure_ascii=False, indent=4)