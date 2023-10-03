# -*- coding:utf-8 -*-
# @Time      :2023/8/8 11:13
# @FileName  :last_column.py
# @Author    :GDJ
# @Software  :PyCharm


import csv
import numpy as np

# 读取CSV文件1
csv_file1 = '../data/restaurant/restaurant_data.csv'
csv_file2 = 'truth_val.csv'

last_column_values = []

# 读取 CSV 文件并获取最后一列的值
with open(csv_file1, 'r') as csv_file:
    reader = csv.reader(csv_file)

    for row in reader:
        last_column_value = row[-1]  # 获取最后一列的值
        last_column_values.append(int(last_column_value))

# 去重并按原顺序输出最后一列的值
unique_last_column_values = list(set(last_column_values))
unique_last_column_values.sort(key=last_column_values.index)

# 输出结果
print("最后一列的值（去重）:", unique_last_column_values)

l = unique_last_column_values
# print(l)
# 读取 CSV 文件2
data2 = np.genfromtxt(csv_file2, delimiter=',', dtype=int)
new_arr = np.column_stack((data2, l))
# print(new_arr)
np.savetxt(csv_file2, new_arr, delimiter=',', fmt='%d')


