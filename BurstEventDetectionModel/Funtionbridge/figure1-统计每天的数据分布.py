# _*__coding:utf-8 _*__
# @Time :2022/2/24 15:31
# @Author :bay
# @File figure1-统计每天的数据分布.py
# @Software : PyCharm
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import mpl
import datetime

# 这个需要改，已经不管用了
# 改好了 感觉说明不了啥问题
# 微软正黑体：Microsoft JhengHei 微软雅黑体：Microsoft YaHei
# 华文细黑：STXihei  华文新魏：STXinwei 黑体 simhei

# mpl.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
mpl.rcParams['font.sans-serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 10


def read_csv():
    N_times = []
    # names = ['micro_blog', 'know', 'baidu_post_bar']
    datas = pd.read_csv('../corpus/data20220911/test_total_data.csv')
    N_number = []
    times = datas.iloc[:, 1].values
    # print(times)
    n0_time = datetime.datetime.strptime(times[0][:10], '%Y-%m-%d').date()
    print("n10_time", n0_time)
    n1_time = n0_time + datetime.timedelta(days=-1)
    # print(n9_time)
    n2_time = n0_time + datetime.timedelta(days=-2)
    # print(n8_time)
    n3_time = n0_time + datetime.timedelta(days=-3)
    # print(n7_time)
    n4_time = n0_time + datetime.timedelta(days=-4)
    # print(n6_time)
    n5_time = n0_time + datetime.timedelta(days=-5)
    # print(n5_time)
    n6_time = n0_time + datetime.timedelta(days=-6)
    # print(n4_time)
    n7_time = n0_time + datetime.timedelta(days=-7)
    # print(n3_time)
    n8_time = n0_time + datetime.timedelta(days=-8)
    # print(n2_time)
    n9_time = n0_time + datetime.timedelta(days=-9)
    n10_time = n0_time + datetime.timedelta(days=-10)
    # print(n9_time)
    n11_time = n0_time + datetime.timedelta(days=-11)
    # print(n8_time)
    n12_time = n0_time + datetime.timedelta(days=-12)
    # print(n7_time)
    n13_time = n0_time + datetime.timedelta(days=-13)
    # print(n6_time)
    n14_time = n0_time + datetime.timedelta(days=-14)
    # print(n5_time)
    n15_time = n0_time + datetime.timedelta(days=-15)
    # print(n4_time)
    n16_time = n0_time + datetime.timedelta(days=-16)
    # print(n3_time)
    n17_time = n0_time + datetime.timedelta(days=-17)
    # print(n2_time)
    n18_time = n0_time + datetime.timedelta(days=-18)
    n19_time = n0_time + datetime.timedelta(days=-19)
    # print(n2_time)
    n20_time = n0_time + datetime.timedelta(days=-20)
    # print(n1_time)
    N_times.append(n20_time.strftime('%Y-%m-%d'))
    N_times.append(n19_time.strftime('%Y-%m-%d'))
    N_times.append(n18_time.strftime('%Y-%m-%d'))
    N_times.append(n17_time.strftime('%Y-%m-%d'))
    N_times.append(n16_time.strftime('%Y-%m-%d'))
    N_times.append(n15_time.strftime('%Y-%m-%d'))
    N_times.append(n14_time.strftime('%Y-%m-%d'))
    N_times.append(n13_time.strftime('%Y-%m-%d'))
    N_times.append(n12_time.strftime('%Y-%m-%d'))
    N_times.append(n11_time.strftime('%Y-%m-%d'))
    N_times.append(n10_time.strftime('%Y-%m-%d'))
    N_times.append(n9_time.strftime('%Y-%m-%d'))
    N_times.append(n8_time.strftime('%Y-%m-%d'))
    N_times.append(n7_time.strftime('%Y-%m-%d'))
    N_times.append(n6_time.strftime('%Y-%m-%d'))
    N_times.append(n5_time.strftime('%Y-%m-%d'))
    N_times.append(n4_time.strftime('%Y-%m-%d'))
    N_times.append(n3_time.strftime('%Y-%m-%d'))
    N_times.append(n2_time.strftime('%Y-%m-%d'))
    N_times.append(n1_time.strftime('%Y-%m-%d'))
    N_times.append(n0_time.strftime('%Y-%m-%d'))
    # print(n1_time, n2_time, n3_time, n4_time, n5_time, n6_time, n7_time, n8_time)
    n20 = n19 = n18 = n17 = n16 = n15 = n14 = n13 = n12 = n11 = n0 =\
        n1 = n2 = n3 = n4 = n5 = n6 = n7 = n8 = n9 = n10 = 0
    for time in times:
        time = datetime.datetime.strptime(time[:10], '%Y-%m-%d').date()
        if time == n20_time:
            n0 += 1
        if time == n19_time:
            n1 += 1
        if time == n18_time:
            n2 += 1
        if time == n17_time:
            n3 += 1
        if time == n16_time:
            n4 += 1
        if time == n15_time:
            n5 += 1
        if time == n14_time:
            n6 += 1
        if time == n13_time:
            n7 += 1
        if time == n12_time:
            n8 += 1
        if time == n11_time:
            n9 += 1
        if time == n10_time:
            n10 += 1
        if time == n9_time:
            n11 += 1
        if time == n8_time:
            n12 += 1
        if time == n7_time:
            n13 += 1
        if time == n6_time:
            n14 += 1
        if time == n5_time:
            n15 += 1
        if time == n4_time:
            n16 += 1
        if time == n3_time:
            n17 += 1
        if time == n2_time:
            n18 += 1
        if time == n1_time:
            n19 += 1
        if time == n0_time:
            n20 += 1
    N_number.append(n0)
    N_number.append(n1)
    N_number.append(n2)
    N_number.append(n3)
    N_number.append(n4)
    N_number.append(n5)
    N_number.append(n6)
    N_number.append(n7)
    N_number.append(n8)
    N_number.append(n9)
    N_number.append(n10)
    N_number.append(n11)
    N_number.append(n12)
    N_number.append(n13)
    N_number.append(n14)
    N_number.append(n15)
    N_number.append(n16)
    N_number.append(n17)
    N_number.append(n18)
    N_number.append(n19)
    N_number.append(n20)

    return N_times, N_number


def plt_bar(N_times, N_number):
    # 不显示框线
    # fig, ax = plt.subplots()
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    x = N_times
    y = N_number
    # plt.plot(x, y)
    #  alpha=0.5
    plt.barh(x, y, color='#6495ED',)  # 绘制横向柱状图
    # plt.bar(x, y, color='blue', width=0.5)
    # X刻度变小
    # plt.tick_params(axis='y', labelsize=9)  # 设置x轴标签大小
    # plt.xticks(x, x, rotation=25)

    plt.xlabel('number of posts/items', fontsize=14)
    # plt.ylabel('time')
    plt.savefig(r'D:\workspace\pycharm\PaperTrail\Funtionbridge\result_img\figure2-数据分布.png')
    plt.show()


def plt_cdf(arr):
    # TODO 查看数据分布情况
    # plt.subplot()
    hist1, bin_edges1 = np.histogram(arr[:10])
    # hist2, bin_edges2 = np.histogram(arr[8:15])
    # hist3, bin_edges3 = np.histogram(arr[16:24])
    cdf1 = np.cumsum(hist1 / sum(hist1))
    # cdf2 = np.cumsum(hist2 / sum(hist2))
    # cdf3 = np.cumsum(hist3 / sum(hist3))
    plt.title('CDF曲线')
    plt.plot(bin_edges1[1:], cdf1, '-^', color='#00FF7F', label='微博')   # 沙棕色
    # plt.plot(bin_edges2[1:], cdf2, '-o', color='#FFA500', label='知乎')   # 适合春天的绿色
    # plt.plot(bin_edges3[1:], cdf3, '-s', color='#87CEEB', label='百度贴吧')   # 天蓝色
    plt.legend(loc='best')
    plt.xlabel('爬取的社交媒体发布的信息（条数/天）')
    plt.show()
# N_number = read_csv()
# print(N_number)
# plt_cdf(N_number)


if __name__ == '__main__':
    N_times, N_number = read_csv()
    print(N_times, N_number)
    plt_bar(N_times, N_number)



