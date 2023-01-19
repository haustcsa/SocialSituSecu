# _*__coding:utf-8 _*__
# @Time :2022/4/15 0015 15:57
# @Author :bay
# @File SocialMediaInfluenceV1.py
# @Software : PyCharm
import pandas as pd
import json
import datetime
from datetime import timedelta as td
from collections import Counter
import jieba
import jieba.posseg as pseg
import time


def aaray_fenci_word(doc):
    # print(doc)
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
        words = [k for k, flag in pseg.lcut(doc) if k not in stop_words and flag in flags and
                 len(k) > 1]
    return words


# TODO 当时计算这个是咋想的，以后记录一下
# 这个还得在考虑考虑
def get_count_words(tweets_set):
    # 词频统计那块按大小输出来就这么干
    result = []
    for doc in tweets_set:
        # print("doc", doc)
        words = aaray_fenci_word(doc)
        result.append(words)
    print(" result", result)
    # c = Counter()
    # for i in result:
    #     for x in i:
    #         if len(x) > 1 and x != '\r\n':
    #             c[x] += 1
    # # # print("打印分词的结果")
    # result_words = []
    # for (k, v) in c.most_common(n=None):
    #     # print(k, v)
    #     result_words.append(k)
    # print("长度;", len(result_words))
    # # print("result_words；", result_words)
    # return result, result_words


# 判断这些词在哪些文档集中出现过，并计算社交媒体影响力
def compute(si_data, compare_datas, release_codes):
    new_si_data = {}
    for word in si_data:
        si_codes = 0
        for compare_data, release_code in zip(compare_datas, release_codes):
            if word in compare_data:
                # 0.6666487, 0.3332973, 0.3333513
                # print("word", word, "release_code:", release_code)
                if release_code == '1.0':
                    release_code = 0.6666487
                if release_code == '2.0':
                    release_code = 0.3332973
                if release_code == '3.0':
                    release_code = 0.3333513
                # print("修改后的release_code:", release_code, type(release_code))
                si_codes += release_code
        # print("si_codes", si_codes)
        new_si_data[word] = si_codes
    new_si_data = dict(sorted(new_si_data.items(), key=lambda x: x[1], reverse=True))
    # print("new_si_data:\n", new_si_data)
    return new_si_data


# 归一化
def normal(temp_result):
    compare_key1 = list(temp_result)[0]
    compare_key2 = list(temp_result)[-1]
    compare_max = temp_result[compare_key1]
    compare_min = temp_result[compare_key2]
    for key in temp_result.keys():
        temp_result[key] = round((temp_result[key] - compare_min) / (compare_max - compare_min), 7)
    return temp_result


def social_data():
    news_data = pd.read_csv('../../corpus/test_mini_data_process.csv').astype(str)
    # news_data = pd.read_csv('../corpus/data20220911/test_total_data_process2.csv').astype(str)
    # TODO 1、按天计算
    # 取出当天的数据
    news_data['time'] = news_data['release_time'].apply(lambda x: str(x)[0:10])
    start_time = datetime.date(*map(int, news_data['time'].min().split('-')))
    end_time = datetime.date(*map(int, news_data['time'].max().split('-')))
    time_cha = (end_time - start_time).days + 1
    for time_i in range(1, time_cha):
        _time1 = (start_time + td(days=time_i)).strftime("%Y-%m-%d")
        print("当前的时间", _time1)
        temp_data = news_data[news_data['time'] == _time1]
        # 取出当天的数据
        tweets_set = temp_data['content_check'].values.tolist()
        # 一天的数据
        get_count_words(tweets_set)
        # compare_datas, result_words = get_count_words(tweets_set)
        # print(len(compare_datas))
        # # # TODO 这个还得理一理
        # release_codes = temp_data['release_source_code'].values.tolist()
        # # print(len(release_codes))
        # social_words = compute(result_words, compare_datas, release_codes)
        # temp_result = normal(social_words)
        # file_path = '../results/word_social/' + _time1 + '_social.json'
        # with open(file_path, 'w', encoding='utf-8') as f:
        #     json.dump(temp_result, f, ensure_ascii=False, indent=4)
        # print(file_path + "存储完成")


if __name__ == '__main__':
    start_time = time.perf_counter()
    print(start_time)
    social_data()
    end_time = time.perf_counter()
    print(end_time)
    time_cha = end_time - start_time
    print(time_cha)