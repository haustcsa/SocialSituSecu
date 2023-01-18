# _*__coding:utf-8 _*__
# @Time :2022/9/28 0028 19:57
# @Author :bay
# @File figure3-result_compute_pic.py
# @Software : PyCharm

import matplotlib.pyplot as plt
from pylab import mpl
# mpl.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
mpl.rcParams['font.sans-serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 14


def compute_tri_value():
    # bds = [29, 32, 24, 23]
    # bcs = [20, 18, 19, 19]
    # bds = [38, 39, 41, 41, 42, 41, 42, 43, 44, 40]
    # bcs = [26, 28, 31, 31, 31, 32, 31, 29, 29, 26]

    bds = [20, 21, 23, 25, 26, 26, 28, 30, 31]
    bcs = [17, 17, 18, 21, 21, 22, 22, 20, 20]
    ps = []
    rs = []
    fs = []
    for bd, bc in zip(bds, bcs):
        p = bc/25
        r = bc/bd
        f = (2*p*r)/(p+r)
        ps.append(round(p, 3))
        rs.append(round(r, 3))
        fs.append(round(f, 3))
    print("准确率", ps)
    print("召回率", rs)
    print("F1值", fs)
    return ps, rs, fs


def plot():
    # ps = [0.81818, 0.8, 0.77143, 0.72973, 0.69231, 0.625, 0.625, 0.61538]
    # rs = [0.9, 0.93333, 0.9, 0.9, 0.9, 0.83333, 0.83333, 0.8]
    # f1 = [0.85714, 0.86154, 0.83077, 0.80597, 0.78261, 0.71429, 0.71429, 0.69565]
    # ps = [0.69565, 0.74074, 0.75, 0.78788, 0.81818, 0.8, 0.77143, 0.72973, 0.69231, 0.625]
    # rs = [0.53333, 0.66667, 0.8, 0.86667, 0.9, 0.93333, 0.9, 0.9,  0.9, 0.83333]
    # fs = [0.60377, 0.70175, 0.77419, 0.8254, 0.85714, 0.86154, 0.83077, 0.80597, 0.78261, 0.71429]
    # miu = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    ps = [0.68, 0.68, 0.72, 0.84, 0.84, 0.88, 0.88, 0.8, 0.8]
    rs = [0.85, 0.81, 0.783, 0.84, 0.808, 0.846, 0.786, 0.667, 0.645]
    fs = [0.756, 0.739, 0.75, 0.84, 0.824, 0.863, 0.83, 0.727, 0.714]
    miu = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    # miu = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    plt.plot(miu, ps, marker='o', label='Precision', linewidth=2.5)
    plt.plot(miu, rs, marker='v', label='Recall', linewidth=2.5)
    plt.plot(miu, fs, marker='s', label='F1', linewidth=2.5)
    plt.ylim(0.5, 1)
    # plt.ylabel('突发事件检测效果')
    # /framealpha=0,shadow=False
    plt.xlabel('inter-cluster threshold μ')
    # plt.xlabel('簇间阈值μ')
    # frameon=False
    plt.legend(loc='upper left')
    # plt.savefig(r'D:\workspace\pycharm\PaperTrail\Funtionbridge\result_img\figure3-ch突发词检测效果.png')
    plt.savefig(r'D:\workspace\pycharm\PaperTrail\Funtionbridge\result_img\figure3-en突发词检测效果.png')

    plt.show()


if __name__ == '__main__':
    # ps, rs, fs = compute_tri_value()
    # ps = [0.3333, 0.4, 0.5714, 0.6842, 0.6333, 0.7500, 0.8462, 0.8095, 0.7391, 0.6939, 0.6863]
    # rs = [0.0263, 0.0526, 0.2105, 0.3421, 0.5000, 0.7105, 0.8684, 0.8947, 0.8947, 0.8947, 0.9211]
    # fs = [0.0487, 0.0930, 0.3077, 0.4561, 0.5588, 0.7297, 0.8571, 0.8499, 0.8091, 0.7186, 0.7866]
    # plot(ps, rs, fs)
    plot()