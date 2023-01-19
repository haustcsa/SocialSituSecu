import matplotlib.pyplot as plt  # 绘图
import pandas as pd  # 读取exal文件

plt.rcParams['font.sans-serif'] = ['SimHei']  # 添加黑体作为绘图字体



PIL_style = ["red","white","white","white","orange","blue","green"]

xls = pd.read_excel(r"C:\Users\oOMAOo\Desktop\record_va.xlsx")
plt.xlabel('Epoch')
plt.title('准确率')
for i,li in enumerate(xls):
    data_list = list(xls[li])
    key_map = [i + 1 for i, key in enumerate(data_list)]
    plt.plot(key_map, data_list, color=PIL_style[i], label=li)
    print(key_map)
    print(data_list)

plt.legend()
# plt.savefig('柱状图.png', dip=500)
plt.show()  # 显示方法
