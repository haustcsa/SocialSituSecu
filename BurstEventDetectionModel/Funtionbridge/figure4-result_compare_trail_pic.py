# _*__coding:utf-8 _*__
# @Time :2022/10/3 0003 10:43
# @Author :bay
# @File figure4-result_compare_trail_pic.py
# @Software : PyCharm
import matplotlib.pyplot as plt
from pylab import mpl
import numpy as np

# mpl.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
mpl.rcParams['font.sans-serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 14
#
# list16 = [0.719, 0.767, 0.742]
# list17 = [0.688, 0.733, 0.710]
# list21 = [0.837, 0.878, 0.857]
# list23 = [0.795, 0.853, 0.824]
# list = [0.8, 0.933, 0.862]
list19 = [0.76, 0.792, 0.776]
list21 = [0.8, 0.690, 0.741]
list22 = [0.72, 0.563, 0.632]
list24 = [0.76, 0.823, 0.792]
list = [0.88, 0.846, 0.863]

size = 3
x = np.arange(size)
total_width, n = 0.7, 5
width = total_width / n
x = x - (total_width - width) / 2
print(x)

# miu = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
# 换个色系
# plt.bar(x, list14, width=width, label="文献[14]", color='#87CEEB')
# plt.bar(x+width, list16, width=width, label="文献[16]", color='#87CEFA')
# plt.bar(x+width*2, list17, width=width, label="文献[17]", color='#00BFFF')
# plt.bar(x+width*3, list19, width=width, label="文献[19]", color='#6495ED')
# plt.bar(x+width*4, list, width=width, label="本文", color='#4682B4')
plt.bar(x, list19, width=width, label="Literature[31]", color='blue')
plt.bar(x+width, list21, width=width, label="Literature[33]", color='orange')
plt.bar(x+width*2, list22, width=width, label="Literature[34]", color='green')
plt.bar(x+width*3, list24, width=width, label="Literature[37]", color='red')
plt.bar(x+width*4, list, width=width, label="This paper", color='purple')

plt.xticks(np.arange(3), ('Precision', 'Recall', 'F1'))
plt.ylim(0.0, max(list)+0.2)
# 显示图例
#plt.figure(dpi=300,figsize=(24,24))
# plt.legend(loc='lower right')
# prop={'size': 9}
# 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
plt.legend(loc='upper left', ncol=3, framealpha=0, shadow=False, fontsize='small')
# plt.xlabel("Comparision   Experiments")
# plt.ylabel("Dice  Score")
plt.savefig(r'D:\workspace\pycharm\PaperTrail\Funtionbridge\result_img\figure4-文献对比2', dpi=500)
# 显示柱状图
plt.show()
# plt.plot(miu, acc, marker='o', label='Presion')
# plt.plot(miu, reall, marker='v', label='Recall')
# plt.plot(miu, f1, marker='s', label='F1')
# plt.ylim(0.5, 1)
# plt.ylabel('突发事件检测效果')
# # /framealpha=0,shadow=False
# plt.legend(loc='upper left')
# plt.show()