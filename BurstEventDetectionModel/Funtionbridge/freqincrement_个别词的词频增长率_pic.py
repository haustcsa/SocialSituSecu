# _*__coding:utf-8 _*__
# @Time :2022/5/30 0030 21:47
# @Author :bay
# @File freqincrement高.py
# @Software : PyCharm
# TODO 画图显示个别词的词频增长率
import json
import os
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft JhengHei']


def get_data():
    # 1.准备数据集（500万数据集，先使用14天或者一周的数据，论文中用了10天的数据）
    freq_increment_files = os.listdir('../results/word_increment')
    # print(freq_increment_files)
    # freq_increment_files.sort(key=lambda x: int(x[:x.find("_")]))  # 按照前面的数字字符排序
    new_files = freq_increment_files[10:]
    # print(new_files)
    print(len(new_files))
    new_datas = []
    for new_file in new_files:
        f = open('../results/word_increment/' + new_file, 'r', encoding='utf-8')
        data = json.load(f)
        new_datas.append(data)
    print(len(new_datas))
    # print(new_datas)
    words = ['安倍', '安倍晋三', '胸部', '中枪', '演讲时', '枪击', '身亡']
    result_datas = []
    for word in words:
        result_data = []
        for use_data in new_datas:
            # print(use_data)
            if word in use_data.keys():
                # print(word)
                # print(use_data[word])
                result_data.append(use_data[word])
            else:
                use_data[word] = 0
                # print(use_data[word])
                result_data.append(use_data[word])
        result_datas.append(result_data)
    print(result_datas)
    return words, result_datas


def plot(words, result_datas):
    times = ['2022-07-04', '2022-07-05', '2022-07-06', '2022-07-07', '2022-07-08', '2022-07-09',
             '2022-07-10', '2022-07-11', '2022-07-12', '2022-07-13']
    print(words)
    for result_data, word in zip(result_datas, words):
        plt.plot(times, result_data, label=word)
        plt.xticks(times, times, rotation=25)
        # 'boxoff' 去掉边框

        plt.legend(loc='upper left', ncol=2, framealpha=0, shadow=False)
    plt.show()


if __name__ == '__main__':
    # get_data()
    words, result_datas = get_data()
    plot(words, result_datas)








