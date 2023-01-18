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


def get_word_dict(contents):
    wordSet = set()
    data_list = []
    # 咱先看看sentence是啥
    stop_words = []
    # 确定使用百度停用词库，结合了荆军昌师兄的停用词库
    stop_words_dir = r'D:\workspace\pycharm\PaperTrail\FunctionTrail\baidu_stopwords.txt'
    for stop in open(stop_words_dir, encoding='utf-8'):
        stop_words.append(stop.replace('\n', ''))
    flags = ['a', 'an', 'ad', 'b', 'i', 'j', 'l', 'n', 'nr', 'nrt',
             'ns', 'nt', 'nw', 'nz', 's', 't', 'v', 'vn', 'PEG', 'LOC', 'ORG']
    user_dict_dir = r'D:\workspace\pycharm\PaperTrail\FunctionTrail\user_dict.txt'
    jieba.load_userdict(user_dict_dir)
    for line in contents:
        line = line.replace(" ", "")
        sentence = []
        if len(line) > 5:
            for k, flag in pseg.lcut(line):
                if len(k) > 1 and k not in stop_words and flag in flags:
                    sentence.append(k)
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

    # json_data = json.dumps(new_word_dict)
    # with open(r'../result/worddictionary/newdicts.json', 'w', encoding='utf-8') as f:
    #     json.dump(new_word_dict, f, ensure_ascii=False, indent=4)


def get_tf_idf(json_dict, _time1):
    fa = 0.5
    value_max = max(json_dict.values())
    # print(value_max)
    for key, value in json_dict.items():
        value = fa + fa * value/value_max
        json_dict[key] = round(value, 7)
    # print("计算过后值的权重", json_dict)
    with open(r'D:\workspace\pycharm\PaperTrail\CompareTrail\ZhangElectric\TFIDF_word\{0}_newdicts_dictionary.json'.format(_time1), 'w',
              encoding='utf-8') as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=4)


def get_zhang_tfidf(time_cha, start_time, news_data):
    for time_i in range(1, time_cha):
        # print("变量循环", time_i)
        _time1 = (start_time + td(days=time_i)).strftime("%Y-%m-%d")
        print("当前的日期", _time1)
        temp_data = news_data[news_data['time'] == _time1]
        contents = temp_data['content_check'].values.tolist()
        # get_word_dict(contents)
        json_dict = get_word_dict(contents)
        get_tf_idf(json_dict, _time1)
