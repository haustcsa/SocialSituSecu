# _*__coding:utf-8 _*__
# @Time :2022/10/17 0017 16:04
# @Author :bay
# @File BurstyTerm-Wij.py
# @Software : PyCharm
import os
import pandas as pd
import datetime
from datetime import timedelta as td
import json


def read_files():
    Fij_files = os.listdir(r'D:\workspace\pycharm\PaperTrail\CompareTrail\ZhangqiangCISC\Fij')
    Fij_data = []
    for Fij_file in Fij_files:
        Fij_path = r'D:\workspace\pycharm\PaperTrail\CompareTrail\ZhangqiangCISC\Fij' + '\\' + Fij_file
        # print(Fij_path)
        f = open(Fij_path, 'r', encoding='utf-8')
        data = json.load(f)
        Fij_data.append(data)

    Bij_files = os.listdir(r'D:\workspace\pycharm\PaperTrail\CompareTrail\ZhangqiangCISC\Bij')
    Bij_data = []
    for Bij_file in Bij_files:
        Bij_path = r'D:\workspace\pycharm\PaperTrail\CompareTrail\ZhangqiangCISC\Bij' + '\\' + Bij_file
        # print(Bij_path)
        f = open(Bij_path, 'r', encoding='utf-8')
        data = json.load(f)
        Bij_data.append(data)

    for i_data, j_data in zip(Fij_data, Bij_data):
        for key in i_data.keys():
            if key in j_data.keys():
                i_data[key] = round(0.3 * i_data[key] + 0.7 * j_data[key], 7)
            else:
                i_data[key] = round(i_data[key], 7)
    return Fij_data


def compute_butsty(Fij_data, V):
    print(len(Fij_data))
    new_lists = []
    new_a_list = {}
    for i in range(10):
        T1 = Fij_data[i]
        # print("最初的值", T1)
        # 排序
        T1_update = dict(sorted(T1.items(), key=lambda x: x[1], reverse=True))
        # 每天的关键词
        # print("这个T1_update是个啥", T1_update)
        new_a = {}
        for i, (k, v) in enumerate(T1_update.items()):
            if v >= 0.7236208:
                new_a[k] = v
        # TODO 突发度的判断
        new_a_list.update(new_a)
        new_list = list(new_a)
        new_lists.append(new_list)
    # new_list = list(new_key_list)
    new_a_list = dict(sorted(new_a_list.items(), key=lambda x: x[1], reverse=True))
    print(len(new_a_list))
    print("222 new_a_list", new_a_list)
    file_path = r'D:\workspace\pycharm\PaperTrail\CompareTrail\ZhangqiangCISC\Wij\bursty_value_v{0}.json'.format(V)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(new_a_list, f, ensure_ascii=False, indent=4)
    print("存储完成")
    path = r'D:\workspace\pycharm\PaperTrail\CompareTrail\ZhangqiangCISC\word_zmerge\bursty_v{0}.txt'.format(V)
    # path = r'D:\workspace\pycharm\PaperTrail\CompareTrail\ZhangElectric\TOP100\bursty_v{}.txt'.format(V)
    # path = r'D:\workspace\pycharm\PaperTrail\CompareTrail\_WangComputer\word_zmerge\bursty_v2.txt'
    with open(path, 'w', encoding='utf-8') as f:
        for key in new_lists:
            f.write(" ".join(key))
            f.write('\n')
    f.close()


if __name__ == '__main__':
    V = 1
    Fij_data = read_files()
    print(Fij_data)
    compute_butsty(Fij_data, V)