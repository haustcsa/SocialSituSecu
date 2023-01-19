# _*__coding:utf-8 _*__
# @Time :2022/3/2 15:06
# @Author :bay
# @File code5jieba_fenci.py
# @Software : PyCharm
import jieba.posseg as pseg
import jieba
import datetime
from datetime import timedelta as td
import pandas as pd


# 中科院 院士 潘建伟 量子 科学 量子 科学家 潘建伟 节目 量子 人工智能 当做 生命 量子 量子 飞天
def fenci_word(doc):
    doc = doc.replace(" ", "")
    stop_words = []
    stop_words_dir = r'D:\workspace\pycharm\PaperTrail\FunctionTrail\baidu_stopwords.txt'
    for stop in open(stop_words_dir, encoding='utf-8'):
        stop_words.append(stop.replace('\n', ''))
    flags = ['a', 'an', 'ad', 'b', 'i', 'j', 'l', 'n', 'nr', 'nrt',
             'ns', 'nt', 'nw', 'nz', 's', 't', 'v', 'vn', 'PEG', 'LOC', 'ORG']
    user_dict_dir = r'D:\workspace\pycharm\PaperTrail\FunctionTrail\user_dict.txt'
    jieba.load_userdict(user_dict_dir)
    words = ''
    if len(doc) > 5:

        words = " ".join([k for k, flag in pseg.lcut(doc) if k not in stop_words and flag in flags
                 and len(k) > 1])
    # print("words", words)
    return words


def fenci_data(time_cha, start_time, news_data):
    for time_i in range(0, time_cha):
        # 取出当天的数据
        _time1 = (start_time + td(days=time_i)).strftime("%Y-%m-%d")
        # print("当前的时间", _time1)
        temp_data = news_data[news_data['time'] == _time1]
        contents = temp_data['content_check'].values.tolist()
        result = []
        for content in contents:
            words = fenci_word(content)
            if len(words) > 0:
                result.append(words)
        path = '../results/word_fenci/' + _time1 + '_fenci.txt'
        with open(path, 'w', encoding='utf-8') as f:
            for key in result:
                f.write("".join(key))
                f.write('\n')
        f.close()