# -*--coding:utf-8 -*--
# @Time : 2022/8/23 0023 18:02
# @Author : BAY
# @File : code3TFPDF_train.py
# @Software : PyCharm
import pandas as pd
import datetime
from datetime import timedelta as td
from FunctionTrail.code3TFPDF_model import TFPDF
import json


# 归一化
def normal(temp_result):
    compare_key1 = list(temp_result)[0]
    compare_key2 = list(temp_result)[-1]
    compare_max = temp_result[compare_key1]
    compare_min = temp_result[compare_key2]
    for key in temp_result.keys():
        temp_result[key] = round((temp_result[key] - compare_min) / (compare_max - compare_min), 7)
    return temp_result


def tfpdf_data(time_cha, start_time, news_data):
    # TODO 1、按天计算
    for time_i in range(1, time_cha):
        # 取出当天的数据
        _time1 = (start_time + td(days=time_i)).strftime("%Y-%m-%d")
        print("当前的时间", _time1)
        temp_data = news_data[news_data['time'] == _time1]
        # print("temp_data.shape", temp_data.shape)
        tfpdf = TFPDF(temp_data)
        # tfpdf.get_tags()
        word_weight_dictionary = tfpdf._get_word_weight_dictionary()
        word_weight_dictionary_result = normal(word_weight_dictionary)
        # print(word_weight_dictionary_result)
        file_path = r'D:\workspace\pycharm\PaperTrail\results\word_NTFPDF' + '\\' + _time1 + '_word_weight.json'
        # print(file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(word_weight_dictionary_result, f, ensure_ascii=False, indent=4)
        print("存储完成")