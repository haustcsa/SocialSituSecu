# _*__coding:utf-8 _*__
# @Time :2022/10/17 0017 15:20
# @Author :bay
# @File BurstyTerm_Bij.py
# @Software : PyCharm
import pandas as pd
import jieba
import jieba.posseg as pseg
import json
import datetime
from datetime import timedelta as td
# 计算TFi 过去p时间段内的平均频率


def compute_TFi(_time1):
    wordSet = set()
    data_list = []
    f = open(r'D:\workspace\pycharm\PaperTrail\results\word_fenci' + '\\' + _time1 + '_fenci.txt', encoding='utf-8')
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
    new_word_dict = dict(sorted(wordDict.items(), key=lambda x: int(x[1]), reverse=True))
    # print(new_word_dict)
    return new_word_dict


def sum_dict(a, b):
    temp = dict()
    # python3,dict_keys类似set； | 并集
    for key in a.keys() | b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp


def get_TFi(start_time):
    # python3使用reduce需要先导入
    data_tf = []
    for time_i in range(1, 11):
        _time1 = (start_time + td(days=time_i)).strftime("%Y-%m-%d")
        print("当前的日期", _time1)
        new_word_dict = compute_TFi(_time1)
        data_tf.append(new_word_dict)

    p = len(data_tf)
    print("p的值", p)
    from functools import reduce
    # [a,b,c]列表中的参数可以2个也可以多个，自己尝试。
    result = reduce(sum_dict, [data_tf[0], data_tf[1], data_tf[2], data_tf[3], data_tf[4], data_tf[5],
                                   data_tf[6], data_tf[7], data_tf[8], data_tf[9]])
    # print("合并过后的值：\n", result)
    for key in result.keys():
        result[key] = round(result[key]/p, 7)
    # print("频率的平均值", result)
    # return result
    # 排序 排的的结果不太对的样子, 已改正 是数据类型的原因造成的
    new_result = dict(sorted(result.items(), key=lambda x: float(x[1]), reverse=True))
    print("排序过后的：\n", new_result)
    return new_result


def compute_word_dict(_time1):
    wordSet = set()
    data_list = []
    f = open(r'D:\workspace\pycharm\PaperTrail\results\word_fenci' + '\\' + _time1 + '_fenci.txt', encoding='utf-8')
    for line in f:
        line = line.strip()
        sentence = line.split(' ')
        data_list.append(sentence)
        wordSet.update(sentence)
    wordDict = dict.fromkeys(wordSet, 0)
    for data in data_list:
        for word in data:
            wordDict[word] += 1
    # print("wordDict", wordDict, type(wordDict))
    # 排序
    new_wordDict = dict(sorted(wordDict.items(), key=lambda x: int(x[1]), reverse=True))
    return new_wordDict


def get_TFij(new_result, time_cha):
    for time_i in range(11, time_cha):
        _time1 = (start_time + td(days=time_i)).strftime("%Y-%m-%d")
        print("当前的日期", _time1)
        wordDict = compute_word_dict(_time1)
        for key1 in wordDict.keys():
            if key1 in new_result.keys():
                wordDict[key1] = round((wordDict[key1] - new_result[key1])/wordDict[key1], 7)
            else:
                wordDict[key1] = 1
        new_wordDict = dict(sorted(wordDict.items(), key=lambda x: float(x[1]), reverse=True))
        json_path = r'D:\workspace\pycharm\PaperTrail\CompareTrail\ZhangqiangCISC\Bij\{0}_Bij.json'.format(
            _time1)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(new_wordDict, f, ensure_ascii=False, indent=4)
        print("{}存储完成".format(_time1))


if __name__ == '__main__':
    news_data = pd.read_csv(r'D:\workspace\pycharm\PaperTrail\corpus\data20220911\test_total_data_process2.csv').astype(
        str)
    # TODO 1、按天计算
    # 取出当天的数据
    news_data['time'] = news_data['release_time'].apply(lambda x: str(x)[0:10])
    start_time = datetime.date(*map(int, news_data['time'].min().split('-')))
    end_time = datetime.date(*map(int, news_data['time'].max().split('-')))
    time_cha = (end_time - start_time).days + 1
    new_result = get_TFi(start_time)
    get_TFij(new_result, time_cha)

