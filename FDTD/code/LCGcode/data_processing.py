# -*- coding:utf-8 -*-
# @Time      :2023/5/22 9:29
# @FileName  :data_processing.py
# @Author    :GDJ
# @Software  :PyCharm

import csv
import pandas as pd


def cmp_result(names):
    df = pd.read_csv('../../data/restaurant/restaurant_truth.csv',
                     names=names)
    f2 = open('../../data/restaurant/cmp_result.csv', 'w', newline='')
    write = csv.writer(f2)
    row_num = df.shape[0]
    for i in range(row_num):
        # print('-' * 30, i, df.iloc[i].values.tolist())
        for j in range(i + 1, row_num):
            my_list = [0 if (df.iloc[i][n] == df.iloc[j][n]) else 1 for n in range(6)]
            if 0 in my_list:
                write.writerow(my_list)
    f2.close()


if __name__ == '__main__':
    names = ['name', 'BUILDING', 'STREET', 'ZIPCODE', 'PHONE', 'ID']
    cmp_result(names)
