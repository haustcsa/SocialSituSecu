# _*__coding:utf-8 _*__
# @Time :2022/10/5 0005 17:23
# @Author :bay
# @File figure2-select4_bursty_value_pic.py
# @Software : PyCharm
import matplotlib.pyplot as plt
from pylab import mpl
import numpy as np
from matplotlib.ticker import FuncFormatter

# mpl.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
mpl.rcParams['font.sans-serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 14
# plt.show()
# fig = plt.figure(figsize=(10, 10))
fig = plt.figure()
ax1 = fig.add_subplot(111)
l = [i for i in range(8)]
lx = [0.00060, 0.00050, 0.00040, 0.00030, 0.00020, 0.00010, 0.00009, 0.00008]
b = [89, 92, 93, 94, 98, 105, 105, 105]

plt.bar(l, b, alpha=0.8, color='#6495ED', width=0.5)
for i, (_x, _y) in enumerate(zip(l, b)):
    plt.text(_x-0.1, _y, b[i], color='black', fontsize=12,)  # 将数值显示在图形上
plt.xlabel("the threshold of burst words")
# plt.xlabel('词语的突发度阈值')
# plt.ylabel('包含标记突发词的个数')
plt.ylabel('contains the number of tagged bursts')
plt.xticks(l, lx)

# a = [0.72483, 0.78523,  0.79866, 0.80537, 0.83221, 0.88591, 0.88591, 0.88591]
# a = [0.74615, 0.80769, 0.81538, 0.82308, 0.84615, 0.87692, 0.87692, 0.87692]
a = [0.75424, 0.77966, 0.78814,  0.79661, 0.83051, 0.88983, 0.88983, 0.88983]
a = [round(item*100, 4) for item in a]
print(a)
ax2 = ax1.twinx()  # this is the important function
# lx = [0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001, 0.00009, 0.00008]
ax2.plot(l, a, 'blue', linewidth=2.5, marker="*", markersize=10)
# plt.xlabel("词语的突发度阈值")

# plt.ylabel("突发词和标记突发词的比值")
plt.ylabel("the ratio of the burst word to the tagged burst word")


def to_percent(temp, position):
    print(temp)
    # return '%1.0f'%(temp) + '%'
    return '%.0f'%(temp) + '%'


plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))

for i, (_x, _y) in enumerate(zip(l, a)):
    plt.text(_x-0.4, _y-0.6, str(a[i])+'%', color='black', fontsize=12,)  # 将数值显示在图形上
# # ax1.legend(loc=1)
# plt.show()
# # ax2.legend(loc=2)
# # ax2.set_ylim([0, 2500])  #设置y轴取值范围
# [132, 132, 132, 124, 120, 119, 117, 108]
# b = [108, 117, 119, 120, 124, 132, 132, 132]

# plt.legend(prop={'family':'SimHei','size':8},loc="upper left")


# xlable为啥不显示


plt.savefig(r'D:\workspace\pycharm\PaperTrail\Funtionbridge\result_img\figure2-en词语突发度阈值v3.png', dpi=500)
plt.show()
