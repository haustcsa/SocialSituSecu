# _*__coding:utf-8 _*__
# @Time :2022/5/12 0012 10:55
# @Author :bay
# @File 1、TFIDFimpro.py
# @Software : PyCharm
import pandas as pd
import json
import datetime
from datetime import timedelta as td
# TODO 这个文件正式使用， 不在执行重复分词


def get_word_dict(_time1):
    wordSet = set()
    data_list = []
    f = open(r'D:\workspace\pycharm\PaperTrail\results\word_fenci' + '\\' + _time1 + '_fenci.txt', encoding='utf-8')
    for line in f:
        line = line.strip()
        sentence = line.split(' ')
        # print("sntence", sentence)
        data_list.append(sentence)

        wordSet.update(sentence)
    # Dj是每一天的数据总数
    Dj = len(data_list)
    # print("每一天的数据总数", Dj)
    wordDict = dict.fromkeys(wordSet, 0)

    for data in data_list:
        for word in data:
            wordDict[word] += 1
    # print("wordDict", wordDict, type(wordDict))
    # 排序

    new_word_dict = dict(sorted(wordDict.items(), key=lambda x: int(x[1]), reverse=True))
    # print("new_word_dict:\n", new_word_dict)
    # sntenceDict是单词Wi在数据集Dj中出现的次数
    sentenceDict = {}
    for key in new_word_dict.keys():
        flag = 0
        for data in data_list:
            if key in data:
                flag += 1
                sentenceDict[key] = flag/Dj
    # print("sntenceDict:\n", sentenceDict)
    return new_word_dict, sentenceDict


def get_tf_idf(wordDict, sentenceDict, _time1):
    fa = 0.4
    value_max = max(wordDict.values())
    # print(value_max)
    for key1, key2 in zip(sentenceDict.keys(), wordDict.keys()):
        # print(key1, dict_map[key1], key2, dict_map1[key2])
        # print("*************************")
        # print(sentenceDict[key1], wordDict[key2])
        value = fa * sentenceDict[key1] + (1-fa) * wordDict[key2]/value_max
        wordDict[key2] = round(value, 7)
    # print("计算过后值的权重", json_dict)
    new_wordDict = dict(sorted(wordDict.items(), key=lambda x: float(x[1]), reverse=True))
    json_path = r'D:\workspace\pycharm\PaperTrail\CompareTrail\ZhangqiangCISC\Fij\{0}_newdicts_dictionary.json'.format(_time1)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(new_wordDict, f, ensure_ascii=False, indent=4)
    print("{}存储完成".format(_time1))


def get_zhang_tfidf(time_cha, start_time):
    for time_i in range(11, time_cha):
        _time1 = (start_time + td(days=time_i)).strftime("%Y-%m-%d")
        print("当前的日期", _time1)
        wordDict, sentenceDict = get_word_dict(_time1)
        get_tf_idf(wordDict, sentenceDict, _time1)


if __name__ == '__main__':
    news_data = pd.read_csv(r'D:\workspace\pycharm\PaperTrail\corpus\data20220911\test_total_data_process2.csv').astype(str)
    # TODO 1、按天计算
    # 取出当天的数据
    news_data['time'] = news_data['release_time'].apply(lambda x: str(x)[0:10])
    start_time = datetime.date(*map(int, news_data['time'].min().split('-')))
    end_time = datetime.date(*map(int, news_data['time'].max().split('-')))
    time_cha = (end_time - start_time).days + 1
    print(start_time, time_cha)
    get_zhang_tfidf(time_cha, start_time)

    # get_a1(get_df())