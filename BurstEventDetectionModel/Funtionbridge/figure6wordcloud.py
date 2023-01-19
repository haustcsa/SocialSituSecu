# _*__coding:utf-8 _*__
# @Time :2022/11/7 0007 20:08
# @Author :bay
# @File figure6wordcloud.py
# @Software : PyCharm
import jieba
import re
from collections import Counter
import random
import pyodbc
from pyecharts import options as opts
from pyecharts.charts import WordCloud
import pandas as pd
import json


def get_data():
    result_data = []
    f = open(r'D:\workspace\pycharm\PaperTrail\results\word_zmerge_value\bursty_value_v8.json', encoding='utf-8')
    data = json.load(f)
    # result_data.append(data)
    # print(result_data)
    # print(data)
    my_list = list(zip(data.keys(), data.values()))[:100]
    print(my_list)
    print(len(my_list))
# [("乌克兰", 12),("俄罗斯", 10)]
    return my_list


def use_pyecharts(rs):
    c = (
            WordCloud()
            .add(series_name="可以写成查询的那个词(俄罗斯)",
                 data_pair=rs,
                 word_size_range=[10, 40],
                 # mask_image='中国地图-solid.png',
                 shape="cursive",
                 textstyle_opts=opts.TextStyleOpts(
                     font_family="cursive",
                     # font_family='MicrosoftYaHei',
                     font_weight='bold',
                 ))
            .set_global_opts(title_opts=opts.TitleOpts(
                title="俄罗斯的相关词云图",
                title_textstyle_opts=opts.TextStyleOpts(
                    font_size=20,
                    color='black'
                )))
            .render("luodun_wordcloud4.html")
    )

    print("成功生成词云图")


if __name__ == '__main__':
    my_list = [('下架', 0.4480298), ('约翰逊', 0.362117), ('宣布', 0.2392319), ('辞职', 0.2355248), ('用户', 0.230925), ('武汉大学', 0.1856392), ('余秀华', 0.1520492), ('苹果', 0.1337179), ('删除', 0.1219669), ('南京', 0.1144469), ('超生', 0.1), ('安倍', 0.1), ('雨衣男', 0.1), ('华为', 0.0989889), ('警方', 0.087775), ('谭谈交通', 0.0772994), ('统一', 0.0643181), ('唐山', 0.0640566), ('霍乱病例', 0.0571685), ('家暴', 0.0565446), ('全州', 0.0517377), ('母女', 0.0352668), ('豆瓣', 0.0312354), ('学生', 0.0309288), ('女孩', 0.0308099), ('终止', 0.0249176), ('发射', 0.0223173), ('女大学生', 0.0214272), ('男子', 0.0211177), ('嫌疑人', 0.0190329), ('违规', 0.0173059), ('死刑', 0.0167398), ('一例', 0.016577), ('总监', 0.0161587), ('招聘', 0.0156313), ('集团', 0.0152067), ('涉嫌', 0.0142263), ('失去', 0.0139932), ('两人', 0.0136235), ('数据库', 0.013552), ('乐视', 0.01336), ('本地文件', 0.0126824), ('违法犯罪', 0.0114119), ('侵权', 0.0109837), ('特斯拉', 0.0106555), ('开启', 0.0105563), ('英国首相', 0.0100711), ('美元', 0.0099518), ('街道', 0.0097325), ('内卷', 0.0095385), ('紧急状态', 0.0090972), ('社会调剂', 0.0085268), ('子女', 0.0079346), ('永久', 0.0078894), ('总统', 0.0078473), ('预约', 0.0076549), ('3亿', 0.0076467), ('云南', 0.007352), ('旅游', 0.007171), ('被封', 0.0071394), ('10亿', 0.0063449), ('身亡', 0.0061967), ('苏州', 0.0061922), ('冰淇淋', 0.0061295), ('迪士尼', 0.0055404), ('袁冰妍', 0.0054696), ('致歉', 0.005423), ('画面', 0.0054199), ('宣判', 0.0052635), ('开分', 0.0051743), ('起火', 0.0050473), ('被判', 0.0049505), ('市政府', 0.0048864), ('世茂', 0.0047002), ('谭乔', 0.0044465), ('陈春花', 0.0044069), ('总理', 0.0042382), ('一审', 0.0041783), ('新东方', 0.0040783), ('北京', 0.0037469), ('抱走', 0.0037314), ('刺杀', 0.0036934), ('米老鼠', 0.0036483), ('调剂', 0.0035074), ('广西', 0.0034073), ('小树林', 0.0033813), ('广播', 0.0032519), ('被家暴', 0.0032251), ('局长', 0.0032053), ('比亚迪', 0.0031898), ('清算', 0.0030503), ('印度', 0.0030267), ('拘留', 0.003005), ('美国', 0.0029596)]
    use_pyecharts(my_list)
