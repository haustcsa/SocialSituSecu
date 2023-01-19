# _*__coding:utf-8 _*__
# @Time :2022/4/15 0015 15:57
# @Author :bay
# @File SocialMediaInfluenceV1.py
# @Software : PyCharm

import numpy as np
import pandas as pd
import operator
import math

"""
  函数功能计算社交媒体影响力
  Rewards 转发数,Comments 评论数,Likes 点赞数,
  Type 社交媒体类型,微博 1，知乎 0.7，百度贴吧 0.5
  Updates 更新数
 返回值：社交媒体影响力
"""


def get_website():
    # 知乎、微博
    R_alexas = [42, 19, 4]
    W_baidus = [9, 9, 9]
    V_PRs = [9, 6, 8]
    V_RLs = [526748, 22125, 29081]
    W_sites = []
    for R_alexa, W_baidu, V_PR, V_RL in zip(R_alexas, W_baidus, V_PRs, V_RLs):
        W_site = (V_RL*(W_baidu+V_PR))/R_alexa
        W_sites.append(W_site)
    return W_sites


def get_SI_data(SI_data):
    Updates = SI_data.shape[0]
    print(Updates)
    S_midias = []
    Weibo_Rewards = Weibo_Comments = Weibo_Likes = Weibo_Updates = 0
    Zhihu_Rewards = Zhihu_Comments = zhihu_Likes = Zhihu_Updates = 0
    Baidu_Rewards = Baidu_Comments = Baidu_Likes = Baidu_Updates = 0
    for index, row in SI_data.iterrows():
        if row['release_source_code'] == 1.0:
            # print("微博")
            Weibo_Rewards += row['rewards']
            Weibo_Comments += row['comments']
            Weibo_Likes += row['likes']
            Weibo_Updates += 1
        if row['release_source_code'] == 2.0:
            # print("知乎")
            Zhihu_Rewards += row['rewards']
            Zhihu_Comments += row['comments']
            zhihu_Likes += row['likes']
            Zhihu_Updates += 1
        if row['release_source_code'] == 3.0:
            Baidu_Rewards += row['rewards']
            Baidu_Comments += row['comments']
            Baidu_Likes += row['likes']
            Baidu_Updates += 1
    # print("微博的量", Weibo_Rewards, Weibo_Comments, Weibo_Likes)
    # print("知乎的量", Zhihu_Rewards, Zhihu_Comments, zhihu_Likes)
    # print("百度的量", Baidu_Rewards, Baidu_Comments, Baidu_Likes)

    Weibo_SI = ((Weibo_Rewards + Weibo_Comments + Weibo_Likes)/Weibo_Updates) / Updates
    Zhihu_SI = ((Zhihu_Rewards + Zhihu_Comments + zhihu_Likes)/Zhihu_Updates) / Updates
    Badu_SI = ((Baidu_Rewards + Baidu_Comments + Baidu_Likes)/Baidu_Updates) / Updates
    print("使用公式1计算出来的社交媒体影响力（SI）：\n", Weibo_SI, Zhihu_SI, Badu_SI)
    S_midias.append(Weibo_SI)
    S_midias.append(Zhihu_SI)
    S_midias.append(Badu_SI)
    return S_midias


def compute_influence(S_midias, W_sites):
    S_influences = []
    for S_midia, W_site in zip(S_midias, W_sites):
        if S_midia > 0:
            S_influence = S_midia*W_site
        else:
            S_influence = W_site/1000
        S_influences.append(S_influence)
    return S_influences


# 均值归一化
def normal(data):
    average = float(sum(data)) / len(data)
    data_1 = [round(abs(x - average) / (max(data) - min(data)), 7) for x in data]
    return data_1


# def softmax(data):
#     """
#     非线性映射归一化函数。归一化到[0, 1]区间，且和为1。归一化后的数据列依然保持原数据列中的大小顺序。
#     非线性函数使用以e为底的指数函数:math.exp()。
#     使用它可以把输入数据的范围区间（-∞, +∞）映射到（0, +∞），这样就可以使得该函数有能力处理负数。
#
#     :param data: 数据列，数据的取值范围是全体实数
#     :return:
#     """
#     exp_list = [math.exp(i) for i in data]
#     sum_exp = sum(exp_list)
#     new_list = []
#     for i in exp_list:
#         new_list.append(i / sum_exp)
#     return new_list


if __name__ == '__main__':
    W_sites = get_website()
    SI_data = pd.read_csv('../corpus/data20220911/test_total_data_process2.csv', encoding='utf-8')
    # # process_data()
    S_midias = get_SI_data(SI_data)
    # print("归一化后的社交媒体影响力（SI）：", N_Weibo_SI, N_Zhihu_SI, N_Baidu_SI)
    # print("归一化后的社交媒体影响力（SI）小数位保留三位：", round(N_Weibo_SI, 7), round(N_Zhihu_SI, 7), round(N_Baidu_SI, 7))
    S_influnces = compute_influence(S_midias, W_sites)
    print("最后计算的结果", S_influnces)

    result_data = normal(S_influnces)
    print(result_data)
    # result_data2 = softmax(result_data)
    # print(result_data2)
