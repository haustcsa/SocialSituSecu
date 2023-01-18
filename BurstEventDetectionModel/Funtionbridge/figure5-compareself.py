# _*__coding:utf-8 _*__
# @Time :2022/10/23 0023 16:26
# @Author :bay
# @File figure5-compareself.py
# @Software : PyCharm
import matplotlib.pyplot as plt
from pylab import mpl
import numpy as np
# mpl.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
mpl.rcParams['font.sans-serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 14


def plot():
    st = [0.72, 0.563, 0.632]
    spt = [0.88, 0.815, 0.846]
    ti = [0.72, 0.529, 0.610]
    pti = [0.88, 0.815, 0.846]
    sti = [0.8, 0.690, 0.740]
    spti = [0.88, 0.845, 0.863]
    size = 3
    x = np.arange(size)
    total_width, n = 0.7, 5
    width = total_width / n
    x = x - (total_width - width) / 2
    print(x)
    plt.bar(x, st, width=width, label="S_T", hatch='/', color='#34B3F1', edgecolor="k")
    plt.bar(x + width, spt, width=width, label="S_PT", hatch='/', color='#FBCB0A', edgecolor="k")
    plt.bar(x + width * 2, ti, width=width, label="T_I", hatch='-', color='#5FD068', edgecolor="k")
    plt.bar(x + width * 3, pti, width=width, label="PT_I", hatch='x', color='#F9CEEE', edgecolor="k")
    plt.bar(x + width * 4, sti, width=width, label="S_T_I", hatch='+', color='#069a8e', edgecolor="k")
    plt.bar(x + width * 5, spti, width=width, label="S_PT_I", hatch='.', color='#e6ba95', edgecolor="k")

    plt.xticks(np.arange(3), ('Precision', 'Recall', 'F1'))
    plt.ylim(0.5, 1)
    # plt.ylim(0.0, max(list) + 0.1)
    # framealpha=0
    # 显示图例 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    plt.legend(loc='best', ncol=3, shadow=True, fontsize='small')
    plt.savefig(r'D:\workspace\pycharm\PaperTrail\Funtionbridge\result_img\figure5-消融对比2', dpi=500)
    plt.show()


if __name__ == '__main__':
    # ps = [0.3333, 0.4, 0.5714, 0.6842, 0.6333, 0.7500, 0.8462, 0.8095, 0.7391, 0.6939, 0.6863]
    # rs = [0.0263, 0.0526, 0.2105, 0.3421, 0.5000, 0.7105, 0.8684, 0.8947, 0.8947, 0.8947, 0.9211]
    # fs = [0.0487, 0.0930, 0.3077, 0.4561, 0.5588, 0.7297, 0.8571, 0.8499, 0.8091, 0.7186, 0.7866]
    plot()
