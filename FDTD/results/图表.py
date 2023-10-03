import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 创建数据
data = [[1, -0.1247, -0.57612, -0.569975, 0.00051],
        [-0.1247, 1, -0.005193, 0.007191, 0.081637],
        [-0.57612, -0.005193, 1, -0.004249, 0.001572],
        [-0.569975, 0.007191, -0.004249, 1, 0.021695],
        [0.00051, 0.081637, 0.001572, 0.021695, 1]]

# 创建DataFrame
df = pd.DataFrame(data, columns=['RN', 'BN', 'SN', 'ZC', 'PN'], index=['RN', 'BN', 'SN', 'ZC', 'PN'])
color_map = plt.cm.colors.LinearSegmentedColormap.from_list(
    'custom_map', [(47/255, 117/255, 181/255),
                   (221/255, 235/255, 247/255),
                   (255/255, 80/255, 80/255)])

# 绘制热力图
plt.figure(figsize=(8, 6))
# 设置图形大小
sns.heatmap(df, cmap=color_map, annot=True, fmt=".3f")

# 设置轴标签
plt.xlabel('X Label')
plt.ylabel('Y Label')

# 显示图表
plt.show()