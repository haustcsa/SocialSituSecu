# _*__coding:utf-8 _*__
# @Time :2022/4/15 0015 15:57
# @Author :bay
# @File SocialMediaInfluenceV1.py
# @Software : PyCharm

import numpy as np
import pandas as pd

"""
  函数功能计算社交媒体影响力
  Rewards 转发数,Comments 评论数,Likes 点赞数,
  Type 社交媒体类型,微博 1，知乎 0.7，百度贴吧 0.5
  Updates 更新数
 返回值：社交媒体影响力
"""


def get_SI_data(SI_data):
    Updates = SI_data.shape[0]
    print(Updates)
    Weibo_Rewards = Weibo_Comments = Weibo_Likes = Weibo_Updates = 0
    Zhihu_Rewards = Zhihu_Comments = zhihu_Likes = Zhihu_Updates = 0
    Baidu_Rewards = Baidu_Comments = Baidu_Likes = Baidu_Updates = 0
    for index, row in SI_data.iterrows():
        # print(index, row['release_source'])
        # print(row['release_source_code'])
        if row['release_source_code'] == 1.0:
            # print("微博")
            Weibo_Rewards += row['rewards']
            Weibo_Comments += row['comments']
            Weibo_Likes += row['likes']
            Weibo_Updates += 1
        if row['release_source_code'] == 0.7:
            # print("知乎")
            Zhihu_Rewards += row['rewards']
            Zhihu_Comments += row['comments']
            zhihu_Likes += row['likes']
            Zhihu_Updates += 1
        if row['release_source_code'] == 0.5:
            Baidu_Rewards += row['rewards']
            Baidu_Comments += row['comments']
            Baidu_Likes += row['likes']
            Baidu_Updates += 1
    print("微博的量", Weibo_Rewards, Weibo_Comments, Weibo_Likes)
    print("知乎的量", Zhihu_Rewards, Zhihu_Comments, zhihu_Likes)
    print("百度的量", Baidu_Rewards, Baidu_Comments, Baidu_Likes)

    Weibo_SI = (((Weibo_Rewards + Weibo_Comments + Weibo_Likes)/Weibo_Updates) * 1) / Updates
    Zhihu_SI = (((Zhihu_Rewards + Zhihu_Comments + zhihu_Likes)/Zhihu_Updates) * 0.7) / Updates
    # Badu_SI = (((Baidu_Rewards + Baidu_Comments + Baidu_Likes)/Baidu_Updates) * 0.5) / Updates
    # if Badu_SI == 'nan':
    #     Badu_SI =
    Badu_SI = Zhihu_SI/10
    print("使用公式1计算出来的社交媒体影响力（SI）：\n", Weibo_SI, Zhihu_SI, Badu_SI)
    return Weibo_SI, Zhihu_SI, Badu_SI


# 对SI 进行归一化
def Normal_SI(Weibo_SI, Zhihu_SI, Baidu_SI):
    Sum_SI = Weibo_SI + Zhihu_SI + Baidu_SI
    Weibo_SI = Weibo_SI/Sum_SI
    Zhihu_SI = Zhihu_SI/Sum_SI
    Baidu_SI = Baidu_SI/Sum_SI
    return Weibo_SI, Zhihu_SI, Baidu_SI


def normal_sigmoid(x):
    return 1./(1 + np.exp(-x))


if __name__ == '__main__':
    SI_data = pd.read_csv('../../../../my programs/pycharm/电子学报张仰森/corpus/test67month_totaldata_process.csv', encoding='utf-8')
    # process_data()
    Weibo_SI, Zhihu_SI, Baidu_SI = get_SI_data(SI_data)
    # print(Weibo_SI, Zhihu_SI, Baidu_SI)
    N_Weibo_SI = normal_sigmoid(Weibo_SI)
    # print(N_Weibo_SI)
    N_Zhihu_SI = normal_sigmoid(Zhihu_SI)
    N_Baidu_SI = normal_sigmoid(Baidu_SI)
    print("归一化后的社交媒体影响力（SI）：", N_Weibo_SI, N_Zhihu_SI, N_Baidu_SI)
    print("归一化后的社交媒体影响力（SI）小数位保留三位：", round(N_Weibo_SI, 7), round(N_Zhihu_SI, 7), round(N_Baidu_SI, 7))
