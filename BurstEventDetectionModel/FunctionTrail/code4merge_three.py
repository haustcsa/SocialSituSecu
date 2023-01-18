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


def merge_three(V):
    # 测试数据
    # tfpdf_data = [
    #     {"病例": 1.0,"感染者": 0.8834951,"确诊": 0.8543689,"上海": 0.815534,"新增": 0.7087379,},
    #     {"病例": 1.0, "确诊": 0.949607,"乌克兰": 0.5545076, "核酸检测": 0.5254739,"居住": 0.4952381,},
    #     {"病例": 1.0, "确诊": 0.9324046,"核酸检测": 0.6436969, "诊断": 0.4863026,"企业": 0.4404079,},
    #     {"病例": 1.0,"宇宙": 0.9339154,"确诊": 0.9141044,"美国": 0.8664662, "斯特兰": 0.8493151,},
    #     {"农民工": 1.0,"县城": 0.9240506, "肖战": 0.8661368,"工作": 0.8342691, "人口": 0.6788943,},
    #     {"孩子": 1.0, "信息": 0.7474214, "情况": 0.7381527,"农民工": 0.6817972,"学生": 0.6718716,},
    #     {"秦怡": 1.0, "银行": 0.8820681,"居住": 0.6984127, "新增": 0.669739, "病例": 0.6659564,},
    #     {"俄罗斯": 1.0,  "新增": 0.7543781, "疫情": 0.5963255, "专业": 0.5849867, "感染者": 0.5287433,},
    #     {"美国": 1.0,"工作": 0.8604247, "病例": 0.8486188,"新增": 0.7718119, "台湾": 0.7593241,},
    #     {"美国": 1.0, "人员": 0.5467499, "死亡": 0.4930708, "工作": 0.3851063, "疫情": 0.3508422,},
    #     {"病例": 1.0, "确诊": 0.7995256, "偶像": 0.6717891, "资本": 0.6414348,"美国": 0.6325192,"bay":45},
    #
    #   ]
    # read_files()
    tfpdf_data = read_files()
    compute_butsty(tfpdf_data, V)


if __name__ == '__main__':
    # news_data = pd.read_csv('../corpus/data20220911/test_total_data_process2.csv').astype(str)
    # TODO 1、按天计算
    # 取出当天的数据
    # news_data['time'] = news_data['release_time'].apply(lambda x: str(x)[0:10])
    # start_time = datetime.date(*map(int, news_data['time'].min().split('-')))
    # end_time = datetime.date(*map(int, news_data['time'].max().split('-')))
    # time_cha = (end_time - start_time).days + 1
    merge_three(7)









