# -*--coding:utf-8 -*--
# @Time : 2022/8/31 0031 16:16
# @Author : BAY
# @File : main_tfpdf.py
# @Software : PyCharm
# 改进一下
import time
import datetime
import pandas as pd
from FunctionTrail.code5jieba_fenci import fenci_data
from FunctionTrail.code1SocialMediaInfluenceV3 import social_data
from FunctionTrail.code2freqincrementV2 import increment_data
from FunctionTrail.code3TFPDF_train import tfpdf_data, Ntfpdf_data
from FunctionTrail.code4merge_three import merge_three
from FunctionTrail.code5bursty_similiarity import similiary
from FunctionTrail.code6cluster import cluster

# 测试数据
# news_data = pd.read_csv('../corpus/test_mini_data_process.csv').astype(str)
# 实验数据
news_data = pd.read_csv('../corpus/data20220911/test_total_data_process2.csv').astype(str)
# TODO 1、按天计算
# 取出当天的数据
news_data['time'] = news_data['release_time'].apply(lambda x: str(x)[0:10])
start_time = datetime.date(*map(int, news_data['time'].min().split('-')))
end_time = datetime.date(*map(int, news_data['time'].max().split('-')))
time_cha = (end_time - start_time).days + 1
V = 8
# fenci_data(time_cha, start_time, news_data)
# time.sleep(10)
# social_data(time_cha, start_time, news_data)
# time.sleep(10)
# increment_data(time_cha, start_time)
# time.sleep(10)
# tfpdf_data(time_cha, start_time, news_data)
# Ntfpdf_data(time_cha, start_time, news_data)
# time.sleep(10)
# merge_three(V)
# time.sleep(10)
# similiary(V)
# time.sleep(10)
ts = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
# ts = [1.0, 1.1]
# # ts = [0.7]
# criterion = 'maxclust'
criterion = 'distance'
for t in ts:
    cluster(t, criterion)



