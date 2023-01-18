# -*--coding:utf-8 -*--
# @Time : 2022/7/27 0027 16:04
# @Author : BAY
# @File : select_bursty_value.py
# @Software : PyCharm

import os
import json
import datetime
import pandas as pd


def read_files():
    # 读取词频增长率文件夹中的文件
    increment_files = os.listdir('../results/word_increment')
    # # TODO 果真不按顺序进行输出，所以。。。
    # # increment_files.sort(key=lambda x: int(x[:x.find("_")]))  # 按照前面的数字字符排序
    # print("再次验证文件是不是按顺序的", len(increment_files), increment_files)
    tfpdf_files = os.listdir('../results/word_TFPDF')
    # TODO 对比实验
    # tfpdf_files = os.listdir(r'D:\workspace\pycharm\PaperTrail\CompareTrail\ZhangElectric\TFIDF_word')
    # print("验证文件是不是按顺序的", len(tfpdf_files), tfpdf_files)
    social_files = os.listdir('../results/word_social')
    # print("验证文件是不是按顺序的", len(social_files), social_files)
    # social_files.sort(key=lambda x: int(x[:x.find("_")]))
    # # print("验证文件是不是按顺序的", len(social_files), social_files)
    increment_data = []
    for increment_file in increment_files:
        f = open('../results/word_increment/' + increment_file, 'r', encoding='utf-8')
        data = json.load(f)
        increment_data.append(data)
    tfpdf_data = []
    for tfpdf_file in tfpdf_files:
        f = open('../results/word_TFPDF/' + tfpdf_file, 'r', encoding='utf-8')
        data = json.load(f)
        tfpdf_data.append(data)
    social_data = []
    for social_file in social_files:
        f = open('../results/word_social/' + social_file, 'r', encoding='utf-8')
        data = json.load(f)
        social_data.append(data)

    for i_data, j_data, z_data in zip(tfpdf_data, increment_data, social_data):
        for key in i_data.keys():
            if key in j_data.keys() and key in z_data.keys():
                i_data[key] = i_data[key] * j_data[key] * z_data[key]
            else:
                i_data[key] = i_data[key]
    return tfpdf_data


# 计算突发度
def compute_butsty(tfpdf_data, V, v_list):
    # print(len(tfpdf_data))
    N = 10
    new_lists = []
    new_a_list = {}
    for i in range(N, N+10):
        T1 = tfpdf_data[i]
        # print("最初的值", T1)
        T1_update = dict.fromkeys(T1.keys(), 0)
        # print("所有值赋为0", T1_update)
        for key1 in T1_update.keys():
            # print("key1:", key1)
            for j in range(i-N, i):
                if key1 in tfpdf_data[j].keys():
                    T1_update[key1] += T1[key1] - tfpdf_data[j][key1]
            T1_update[key1] = round(T1_update[key1] / N, 7)
            # print("计算的结果", T1[key1])
        for key1 in T1_update.keys():
            if T1_update[key1] == 0:
                T1_update[key1] = round(T1[key1]/N, 7)
        # 排序
        T1_update = dict(sorted(T1_update.items(), key=lambda x: x[1], reverse=True))
        # 每天的关键词
        # print("这个T1_update是个啥", T1_update)

        new_a = {}
        for i, (k, v) in enumerate(T1_update.items()):
            if v >= v_list:
                new_a[k] = v
        # # TODO 突发度的判断
        new_a_list.update(new_a)
        new_list = list(new_a_list)
        new_lists.append(new_list)
    # new_list = list(new_key_list)
    print("new_a_list", len(new_a_list), new_a_list)
    new_result = list(new_a_list)
    print(len(new_result), new_result)
    # print("new_list", len(new_lists), new_lists)
    file_path = '../results/event_values/{}_event.txt'.format(str(v_list))
    with open(file_path, 'w', encoding='utf-8') as f:
        for word in new_result:
            f.write(word)
            f.write('\n')
    print("存储完成")


def merge_three(V):
    tfpdf_data = read_files()
    v_lists = [0.00008, 0.00009, 0.00010, 0.00020, 0.00030, 0.00040, 0.00050, 0.00060]
    # v_lists = [0.0015]
    for v_list in v_lists:
        compute_butsty(tfpdf_data, V, v_list)


if __name__ == '__main__':
    merge_three(2)









