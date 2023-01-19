# -*--coding:utf-8 -*--
# @Time : 2022/7/21 0021 15:34
# @Author : BAY
# @File : 按天统计词频并画出最高词频的图.py
# @Software : PyCharm
import pandas as pd
import datetime
from datetime import timedelta as td
import jieba
import jieba.posseg as pseg
from collections import Counter
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft JhengHei']


#  输出['看待', '名校', '退出', '国际', '大学排名', '中国人民大学', '南京大学', '国内', '知名', '高校', '退出', '国际', '大学排名']
# def fenci_word(doc):
#     stop_words = []
#     stop_words_dir = r'../FunctionTrail/baidu_stopwords.txt'
#     for stop in open(stop_words_dir, encoding='utf-8'):
#         stop_words.append(stop.replace('\n', ''))
#     flags = ['a', 'an', 'ad', 'b', 'i', 'j', 'l', 'n', 'nr', 'nrt',
#              'ns', 'nt', 'nw', 'nz', 's', 't', 'v', 'vn', 'PEG', 'LOC', 'ORG']
#     user_dict_dir = r'../FunctionTrail/user_dict.txt'
#     jieba.load_userdict(user_dict_dir)
#     words = []
#     try:
#         words = [k for k, flag in pseg.lcut(doc) if k not in stop_words and flag in flags]
#     except:
#         print("'float' object has no attribute 'decode'")
#     return words


def count_words(_time1, k_value):
    result = []
    f = open('../results/word_fenci/' + _time1 + '_fenci.txt', encoding='utf-8')
    for line in f:
        line = line.strip()
        result.append(line.split(' '))
    # result = []
    # for doc in tweets_set:
    #     words = fenci_word(doc)
    #     result.append(words)
    c = Counter()
    for i in result:
        for x in i:
            if len(x) > 1 and x != '\r\n':
                c[x] += 1
    # print("打印分词的结果")
    for (k, v) in c.most_common(100):
        if k == k_value:
            # print(v)
            return v
        # print("%s:%d" % (k, v))


if __name__ == '__main__':
    news_data = pd.read_csv(r'../corpus/data20220911/test_total_data_process2.csv').astype(str)
    # TODO 1、按天计算
    # 取出当天的数据
    time_lists = []
    result_data = []
    news_data['time'] = news_data['release_time'].apply(lambda x: str(x)[0:10])
    start_time = datetime.date(*map(int, news_data['time'].min().split('-')))
    for time_i in range(10, 20):
        # print("循环的变量： ", time_i)
        _time1 = (start_time + td(days=time_i)).strftime("%Y-%m-%d")
        print("当前的时间", _time1)
        time_lists.append(_time1)
        # temp_data = news_data[news_data['time'] == _time1]
        # 取出当天的数据
        # print("目前的数据：\n", temp_data)
        # tweets_set = temp_data['content_check'].values.tolist()
        v = count_words(_time1, "最伟大的作品")
        result_data.append(v)
    x = time_lists
    print(time_lists)
    y = []
    for i in result_data:
        print(i)
        if i == None:
            i = 0
            y.append(i)
        else:
            y.append(i)
    # y = result_data

    print(y)
    plt.bar(x, y, color='blue', width=0.5)
    plt.xticks(x, x, rotation=25)
    # plt.legend()
    plt.title('最伟大的作品')
    plt.savefig('figure3-最伟大的作品.png')
    plt.show()





