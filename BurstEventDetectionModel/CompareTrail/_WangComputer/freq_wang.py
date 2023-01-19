# -*--coding:utf-8 -*--
# @Time : 2022/8/24 0024 10:58
# @Author : BAY
# @File : 1、词频增长率.py
# @Software : PyCharm
import pandas as pd
import json
import jieba
import jieba.posseg as pseg
import datetime
from datetime import timedelta as td
from collections import Counter


def count_words(_time1):
    # 词频统计那块按大小输出来就这么干
    result = []
    f = open('../results/word_fenci/' + _time1 + '_fenci.txt', encoding='utf-8')
    for line in f:
        line = line.strip()
        result.append(line.split(' '))
    c = Counter()
    for i in result:
        for x in i:
            if len(x) > 1 and x != '\r\n':
                c[x] += 1
    # # print("打印分词的结果")
    result_words = {}
    for (k, v) in c.most_common(n=None):
        # print(k, v)
        result_words[k] = v
    print("长度;", len(result_words))
    # print("result_words；", result_words)
    return result_words


def compute_wordfreq(arg1, arg2):
    for key in arg2.keys():
        if key not in arg1.keys():
            arg1[key] = 0
        arg2[key] = round((arg2[key]-arg1[key])/(1+arg1[key]), 7)
    # print("计算过后的arg2:\n", arg2)
    return arg2


# 归一化
def normal(temp_result):
    compare_key1 = list(temp_result)[0]
    compare_key2 = list(temp_result)[-1]
    compare_max = temp_result[compare_key1]
    compare_min = temp_result[compare_key2]
    for key in temp_result.keys():
        temp_result[key] = round((temp_result[key] - compare_min) / (compare_max - compare_min), 7)
    return temp_result


if __name__ == '__main__':
    # 1.准备数据集（500万数据集，先使用14天或者一周的数据，论文中用了10天的数据）
    data = pd.read_csv(r'D:\workspace\pycharm\PaperTrail\corpus\67monthdata\test67month_totaldata_process.csv').astype(str)
    # 2.按时间计算，以天为单位
    result_freqdict = []
    data['time'] = data['release_time'].apply(lambda x: str(x)[0:10])
    start_time = datetime.date(*map(int, data['time'].min().split('-')))
    end_time = datetime.date(*map(int, data['time'].max().split('-')))
    time_cha = (end_time - start_time).days + 1
    # 21
    print("时间差", time_cha)
    for time_i in range(0, time_cha):
        _time1 = (start_time + td(days=time_i)).strftime("%Y-%m-%d")
        print("当前的日期", _time1)
        # 统计第Tn天的词频        # 统计第Tn天的词频
        freqdict = count_words(_time1)
        # print("freqdict", freqdict)
        result_freqdict.append(freqdict)
    for i, j in zip(range(0, 20), range(1, 21)):
        _time = (start_time + td(days=j)).strftime("%Y-%m-%d")
        print(_time)
        # print(i, j)
        # print(result_freqdict[i])
        # print(result_freqdict[j])
        word_freq = compute_wordfreq(result_freqdict[i], result_freqdict[j])
        word_freq = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))
        temp_word_freq = normal(word_freq)
        # file_path = './results/freq/' + _time + '_word_freq.json'
        file_path = r'D:\workspace\pycharm\PaperTrail\CompareTrail\WangComputer\freq_word' + '\\' + _time + '_word_freq.json'
        # print(file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(temp_word_freq, f, ensure_ascii=False, indent=4)
    print("存储完成")

