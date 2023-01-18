# _*__coding:utf-8 _*__
# @Time :2022/5/12 0012 10:55
# @Author :bay
# @File 1、TFIDFimpro.py
# @Software : PyCharm
import pandas as pd
import jieba
import jieba.posseg as pseg
import json
import datetime
from datetime import timedelta as td
# 这个文件需要在改改
# TODO 这个文件正式使用， 不在执行重复分词


def get_word_dict(_time1):
    wordSet = set()
    data_list = []
    f = open('../results/word_fenci/' + _time1 + '_fenci.txt', encoding='utf-8')
    # f = open('../results/word_fenci_test/' + _time1 + '_fenci.txt', encoding='utf-8')
    for line in f:
        line = line.strip()
        sentence = line.split(' ')
        # print("sntence", sentence)
        data_list.append(sentence)
        wordSet.update(sentence)
    wordDict = dict.fromkeys(wordSet, 0)
    for data in data_list:
        for word in data:
            wordDict[word] += 1
    # print("wordDict", wordDict, type(wordDict))
    # 排序
    new_word_dict = dict(sorted(wordDict.items(), key=lambda x: int(x[1]), reverse=True))
    # print("new_word_dict:\n", new_word_dict)
    return new_word_dict


def get_tf_idf(json_dict, _time1):
    fa = 0.5
    value_max = max(json_dict.values())
    # print(value_max)
    for key, value in json_dict.items():
        value = fa + fa * value/value_max
        json_dict[key] = round(value, 7)
    # print("计算过后值的权重", json_dict)
    json_path = r'../CompareTrail/ZhangElectric/TFIDF_word/{0}_newdicts_dictionary.json'.format(_time1)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=4)
    print("存储完成")


def get_zhang_tfidf(time_cha, start_time, news_data):
    for time_i in range(1, time_cha):
        print("变量循环", time_i)
        _time1 = (start_time + td(days=time_i)).strftime("%Y-%m-%d")
        print("当前的日期", _time1)
        json_dict = get_word_dict(_time1)
        get_tf_idf(json_dict, _time1)
