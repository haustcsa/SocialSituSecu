# -*- coding:utf-8 -*-
# @Time      :2023/5/29 10:09
# @FileName  :Source_reliability.py
# @Author    :GDJ
# @Software  :PyCharm

import pandas as pd
import numpy as np
import datetime


def source_score(data):
    l = len(data)
    error_num = 0
    num = 0

    for i in range(l):
        for j in range(i + 1, l):
            num += 1
            if data[i][1] == data[j][1] and data[i][2] == data[j][2] and data[i][3] != data[j][3]:
                error_num += 1
                break

    Error_rate = 1 - (error_num / l)
    return Error_rate


# 按数据源分类
def source_sort(K):
    names = ['source', 'name', 'BUILDING', 'STREET', 'ZIPCODE', 'PHONE', 'ID']
    source_weight = []
    s = 0
    df = pd.read_csv('../data/restaurant/restaurant_data.csv', names=names)
    data = {}

    for k in range(K):
        data[k] = []

    row_num = df.shape[0]

    for i in range(row_num):
        line = df.iloc[i].values.tolist()
        data[line[0]].append(line[1:-1])

    for k in range(K):
        data[k] = source_score(data[k])
        s += data[k]
    # s = sum(data.values())
    # print(s, data)
    for k in range(K):
        source_weight.append(data[k] / s)
    # print(source_weight)
    # print(np.array(source_weight))
    return np.array(source_weight)


if __name__ == '__main__':
    K = 5
    names = ['source', 'name', 'BUILDING', 'STREET', 'ZIPCODE', 'PHONE', 'ID']
    starttime = datetime.datetime.now()
    source_sort(K)
    endtime = datetime.datetime.now()
    print(str((endtime - starttime).seconds))
