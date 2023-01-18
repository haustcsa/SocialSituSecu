# _*__coding:utf-8 _*__
# @Time :2022/9/28 0028 21:53
# @Author :bay
# @File main_socail_tfpdf_increment.py
# @Software : PyCharm

import time
import datetime
import pandas as pd
from FunctionTrail.code5jieba_fenci import fenci_data
from FunctionTrail.code1SocialMediaInfluenceV3 import social_data
from FunctionTrail.code2freqincrementV2 import increment_data
from FunctionTrail.code3TFPDF_train import tfpdf_data
from FuntionCompare.Social_increment.code4merge_three import merge_three
from FuntionCompare.Social_increment.code5bursty_similiarity import similiary
from FuntionCompare.Social_increment.code6cluster import cluster
# 测试数据
# news_data = pd.read_csv('../corpus/test_mini_data_process.csv').astype(str)
# 实验数据
# news_data = pd.read_csv('../corpus/data20220911/test_total_data_process2.csv').astype(str)
# # TODO 1、按天计算
# # 取出当天的数据
# news_data['time'] = news_data['release_time'].apply(lambda x: str(x)[0:10])
# start_time = datetime.date(*map(int, news_data['time'].min().split('-')))
# end_time = datetime.date(*map(int, news_data['time'].max().split('-')))
# time_cha = (end_time - start_time).days + 1
V = 4
# time.sleep(10)
# merge_three(V)
# time.sleep(10)
similiary(V)
time.sleep(10)
ts = [1.2]
# criterion = 'maxclust'
criterion = 'distance'
for t in ts:
    cluster(t, criterion)



