# _*__coding:utf-8 _*__
# @Time :2022/5/30 0030 21:47
# @Author :bay
# @File freqincrement高.py
# @Software : PyCharm
# TODO 词频增长率公式的实现
import json
from datetime import timedelta as td
from collections import Counter
# TODO 这个程序的目的是为了提升代码的运行速度，不用在重复去分词了
# TODO 对比下速度提升 v=212.3782263 v2=0.5334973000000001 速度提升了211.845还是非常有成效的，质的飞跃
# TODO 修改完成，V3版本用于实验


def count_words(_time1):
    result = []
    f = open('../results/word_fenci/' + _time1 + '_fenci.txt', encoding='utf-8')
    for line in f:
        line = line.strip()
        result.append(line.split(' '))
    # print("result", result)
    # 词频统计那块按大小输出来就这么干
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


def increment_data(time_cha, start_time):
    # 2.按时间计算，以天为单位
    result_freqdict = []
    for time_i in range(0, time_cha):
        # print("变量循环", time_i)
        _time1 = (start_time + td(days=time_i)).strftime("%Y-%m-%d")
        print("当前的日期", _time1)
        # 统计第Tn天的词频
        freqdict = count_words(_time1)
        # print("freqdict", freqdict)
        result_freqdict.append(freqdict)
    # print(len(result_freqdict))
    temp_result = get_result(result_freqdict[1], result_freqdict[0], 2)
    result = normal(temp_result)
    save_result(result, '2022-06-24')
    times = ['2022-06-25', '2022-06-26', '2022-06-27', '2022-06-28', '2022-06-29',
             '2022-06-30', '2022-07-01', '2022-07-02', '2022-07-03', '2022-07-04',
             '2022-07-05', '2022-07-06', '2022-07-07', '2022-07-08', '2022-07-09',
             '2022-07-10', '2022-07-11', '2022-07-12', '2022-07-13', ]
    for i, j in zip(range(2, time_cha), times):
        temp_result = get_result(result_freqdict[i], temp_result, i+1)
        result = normal(temp_result)
        save_result(result, j)




