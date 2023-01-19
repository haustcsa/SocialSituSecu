# _*__coding:utf-8 _*__
# @Time :2022/10/14 0014 9:22
# @Author :bay
# @File code2calculationofburstytermscore.py
# @Software : PyCharm
import pandas as pd
from datetime import timedelta as td
import datetime
import json


# 计算词频
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
    return new_word_dict


# 分子  均值+2倍的方差
def get_TFi(start_time):
    # python3使用reduce需要先导入
    data_tf = []
    for time_i in range(1, 11):
        _time1 = (start_time + td(days=time_i)).strftime("%Y-%m-%d")
        print("当前的日期", _time1)
        new_word_dict = compute_TFi(_time1)
        data_tf.append(new_word_dict)
    df = pd.DataFrame([data_tf[0], data_tf[1], data_tf[2], data_tf[3], data_tf[4], data_tf[5],
                                   data_tf[6], data_tf[7], data_tf[8], data_tf[9]])
    df = df.fillna(0)
    mean_result = dict(df.mean())
    var_result = dict(df.var())
    # print(mean_result)
    # print(var_result)
    for k1, k2 in zip(mean_result.keys(), var_result.keys()):
        mean_result[k1] = mean_result[k1] + 2*var_result[k2]
    # print(mean_result)
    return mean_result


def normal(temp_result):
    compare_key1 = list(temp_result)[0]
    compare_key2 = list(temp_result)[-1]
    compare_max = temp_result[compare_key1]
    compare_min = temp_result[compare_key2]
    for key in temp_result.keys():
        temp_result[key] = round((temp_result[key] - compare_min) / (compare_max - compare_min), 7)
    return temp_result


def get_TFij(mean_result, time_cha):
    tags_words = []
    tags_words_dir = r'D:\workspace\pycharm\PaperTrail\CompareTrail\ZhouCIAC\tags.txt'
    for stop in open(tags_words_dir, encoding='utf-8'):
        tags_words.append(stop.replace('\n', ''))
    for time_i in range(11, time_cha):
        _time1 = (start_time + td(days=time_i)).strftime("%Y-%m-%d")
        print("现在的日期", _time1)
        wordDict = compute_TFi(_time1)
        # 这块能不能改进下
        for key1 in wordDict.keys():
            if key1 in tags_words:
                if key1 in mean_result.keys():
                    wordDict[key1] = round((wordDict[key1]/mean_result[key1])**2, 7)
                else:
                    wordDict[key1] = 1
            else:
                if key1 in mean_result.keys():
                    wordDict[key1] = round(wordDict[key1]/mean_result[key1], 7)
                else:
                    wordDict[key1] = 1
        new_wordDict = dict(sorted(wordDict.items(), key=lambda x: float(x[1]), reverse=True))
        temp_data = normal(new_wordDict)
        json_path = r'D:\workspace\pycharm\PaperTrail\CompareTrail\ZhouCIAC\BurstyTermScore\{0}_score.json'.format(
            _time1)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(temp_data, f, ensure_ascii=False, indent=4)
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
    mean_result = get_TFi(start_time)
    get_TFij(mean_result, time_cha)


