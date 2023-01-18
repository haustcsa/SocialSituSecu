# _*__coding:utf-8 _*__
# @Time :2022/4/15 0015 15:57
# @Author :bay
# @File SocialMediaInfluenceV1.py
# @Software : PyCharm
# TODO 这个程序的目的是为了提升代码的运行速度，不用在重复去分词了
# TODO 对比下速度提升 用了一个简单的数据集去测试 V2 = 3.2011919 V3 = 0.047234 速度提升了3.1539579 还是非常有成效的
# TODO 修改完成，V3版本用于实验


from datetime import timedelta as td
from collections import Counter
import json


def get_count_words(_time1):
    # 词频统计那块按大小输出来就这么干
    result = []
    f = open('../results/word_fenci/' + _time1 + '_fenci.txt', encoding='utf-8')
    for line in f:
        line = line.strip()
        result.append(line.split(' '))
    # print("result", result)
    c = Counter()
    for i in result:
        for x in i:
            if len(x) > 1 and x != '\r\n':
                c[x] += 1
    # print("打印分词的结果")
    result_words = []
    for (k, v) in c.most_common(n=None):
        # print(k, v)
        result_words.append(k)
    return result, result_words


# 判断这些词在哪些文档集中出现过，并计算社交媒体影响力
def compute(si_data, compare_datas, release_codes):
    new_si_data = {}
    for word in si_data:
        si_codes = 0
        for compare_data, release_code in zip(compare_datas, release_codes):
            if word in compare_data:
                # 0.6666487, 0.3332973, 0.3333513
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


def social_data(time_cha, start_time,  news_data):
    # # TODO 1、按天计算
    for time_i in range(1, time_cha):
        _time1 = (start_time + td(days=time_i)).strftime("%Y-%m-%d")
        temp_data = news_data[news_data['time'] == _time1]
        print("当前的时间", _time1)
        # 取出当天的数据
        compare_datas, result_words = get_count_words(_time1)
        #  TODO 这个还得理一理
        release_codes = temp_data['release_source_code'].values.tolist()
        # print(len(release_codes))
        social_words = compute(result_words, compare_datas, release_codes)
        temp_result = normal(social_words)
        file_path = '../results/word_social/' + _time1 + '_social.json'
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(temp_result, f, ensure_ascii=False, indent=4)
        print(file_path + "存储完成")