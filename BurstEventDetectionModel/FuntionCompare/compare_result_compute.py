# _*__coding:utf-8 _*__
# @Time :2022/9/28 0028 19:57
# @Author :bay
# @File figure3-result_compute_pic.py
# @Software : PyCharm

import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft JhengHei']


def compute_tri_value():
    # bds = [38, 39, 41, 41, 42, 41, 42, 43, 44, 40]
    # bcs = [26, 28, 31, 31, 31, 32, 31, 29, 29, 26]
    bds = [32, 27,  34, 27, 29, 26]
    bcs = [18, 22, 18, 22, 20, 22]
    ps = []
    rs = []
    fs = []
    for bd, bc in zip(bds, bcs):
        p = bc/25
        r = bc/bd
        f = (2*p*r)/(p+r)
        ps.append(round(p, 5))
        rs.append(round(r, 5))
        fs.append(round(f, 5))
    print("准确率", ps)
    print("召回率", rs)
    print("F1值", fs)
    return ps, rs, fs



def plot(ps, rs, fs):
    # ps = [0.81818, 0.8, 0.77143, 0.72973, 0.69231, 0.625, 0.625, 0.61538]
    # rs = [0.9, 0.93333, 0.9, 0.9, 0.9, 0.83333, 0.83333, 0.8]
    # f1 = [0.85714, 0.86154, 0.83077, 0.80597, 0.78261, 0.71429, 0.71429, 0.69565]
    # ps = [0.69565, 0.74074, 0.75, 0.78788, 0.81818, 0.8, 0.77143, 0.72973, 0.69231, 0.625]
    # rs = [0.53333, 0.66667, 0.8, 0.86667, 0.9, 0.93333, 0.9, 0.9,  0.9, 0.83333]
    # fs = [0.60377, 0.70175, 0.77419, 0.8254, 0.85714, 0.86154, 0.83077, 0.80597, 0.78261, 0.71429]
    # miu = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    miu = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    # miu = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    plt.plot(miu, ps, marker='o', label='Presion', linewidth=2)
    plt.plot(miu, rs, marker='v', label='Recall', linewidth=2)
    plt.plot(miu, fs, marker='s', label='F1', linewidth=2)
    plt.ylim(0.5, 1)
    plt.ylabel('突发事件检测效果')
    # /framealpha=0,shadow=False
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    ps = [0.3333, 0.4, 0.5714, 0.6842, 0.6333, 0.7500, 0.8462, 0.8095, 0.7391, 0.6939, 0.6863]
    rs = [0.0263, 0.0526, 0.2105, 0.3421, 0.5000, 0.7105, 0.8684, 0.8947, 0.8947, 0.8947, 0.9211]
    fs = [0.0487, 0.0930, 0.3077, 0.4561, 0.5588, 0.7297, 0.8571, 0.8499, 0.8091, 0.7186, 0.7866]
    plot(ps, rs, fs)