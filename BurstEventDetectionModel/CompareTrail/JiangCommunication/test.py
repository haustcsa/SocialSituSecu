# -*--coding:utf-8 -*--
# @Time : 2022/9/2 0002 15:47
# @Author : BAY
# @File : test.py
# @Software : PyCharm
from datetime import date, timedelta as td
import datetime
import pandas as pd
# new_dict = {}
# word_dic_tt = {
#     0: {"公司": 0.0022138665497478606,
#         "发射": 0.002139364839834061,
#         "健康": 0.0015031087624443114,
#         "皮肤": 0.001061849942932323,
#         "精神": 0.000828233139649348,},
#     1: {"俄军": 0.0016214825924045773,
#         "信誉": 0.0014472608209391848,
#         "感染者": 0.0010369213874918122,
#         "养老": 0.0010238488971665049,
#         "毕业": 0.0009363544028563952,},
#     2: {"知名": 0.0027214804454652017,
#         "成片": 0.0011225669669040154,
#         "容量": 0.000923083374766815,
#         "本土": 0.0008875856769686972,
#         "确诊": 0.0008762496434990441}
# }
# key_t = {"知名": 0.0027214804454652017,
#         "成片": 0.0011225669669040154,
#         "容量": 0.000923083374766815,
#         "本土": 0.0008875856769686972,
#         "确诊": 0.0008762496434990441}
#
# for k1 in key_t.items():
#     # print(k1)
#     new_dict[k1[0]] = 0
#
#     for k2, values in word_dic_tt.items():
#         # print(k2, values)
#         for k3 in values.items():
#             # print(k3)
#             if k1[0] == k3[0]:
#                 new_dict[k1[0]] += k3[1]
# print("new_dict", new_dict)
news_data = pd.read_csv(r'D:\workspace\pycharm\PaperTrail\corpus\data20220911\test_total_data_process2.csv').astype(str)
# TODO 1、按天计算
# 取出当天的数据
news_data['time'] = news_data['release_time'].apply(lambda x: str(x)[0:10])
start_time = datetime.date(*map(int, news_data['time'].min().split('-')))
temp_time = (start_time + td(days=19)).strftime("%Y-%m-%d")
print(temp_time)