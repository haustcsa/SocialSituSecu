# -*--coding:utf-8 -*--
# @Time : 2022/8/5 0005 16:21
# @Author : BAY
# @File : bursty_similiarity.py
# @Software : PyCharm

import csv
import  os
# 前100个突发词
# 首先得有分词后的文件所在
# 实现共现度


def get_result100(V):
    compare_datas = []
    compare_files = os.listdir(r'../results/word_fenci')
    # print(compare_files)
    for compare_file in compare_files[11:]:
        compare_data = []
        f = open(r'../results/word_fenci/' + compare_file, encoding='utf-8')
        for line in f:
            compare_data.append(line.split(' '))
            # print(compare_data)
        compare_datas.append(compare_data)

    path = r'../CompareTrail/ZhangElectric/TOP100/bursty_v{}.txt'.format(V)
    result100 = open(path, encoding='utf-8')

    datas = []
    for line in result100:
        data = line.split(" ")
        datas.append(data)
    return datas, compare_datas


def compute(data, compare_datas, n):
    # print(data)
    data[-1] = data[-1].split('\n')[0]
    headers = data
    f = open(r'../CompareTrail/ZhangElectric/final_input/' + n + '_input.csv', 'w', encoding='utf8', newline='')
    writer = csv.writer(f)
    # 表头
    writer.writerow(headers)
    rows = []
    for i in range(len(data)):
        # print(data[i])
        row = []
        for j in range(len(data)):
            # print(data[j])
            num1 = 0
            num2 = 0
            num3 = 0
            for compare_data in compare_datas:
                if data[i] in compare_data and data[j] in compare_data:
                    num1 += 1
                if data[j] in compare_data:
                    num2 += 1
                if data[i] in compare_data:
                    num3 += 1
            rowdata1 = num1/(num2+1)
            # print(rowdata1)
            rowdata2 = num1/(num3+1)
            # print(rowdata2)
            rowdata = round(0.5*rowdata1 + 0.5*rowdata2, 7)
            # print(rowdata)
            row.append(rowdata)
        # print(len(row))
        rows.append(row)
        # print(rows)
    # print(len(rows))
    # print(rows)
    writer.writerows(rows)


def similiary(V):
    # get_result100()
    datas, compare_datas = get_result100(V)
    # print(len(datas))
    # print(len(compare_datas))
    times = ['2022-07-04', '2022-07-05', '2022-07-06', '2022-07-07', '2022-07-08', '2022-07-09',
             '2022-07-10', '2022-07-11', '2022-07-12', '2022-07-13']
    for data, compare_data, time in zip(datas, compare_datas, times):

        compute(data, compare_data, time)

