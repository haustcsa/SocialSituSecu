# -*--coding:utf-8 -*--
# @Time : 2022/8/19 0019 15:40
# @Author : BAY
# @File : 层次聚类正式测试.py
# @Software : PyCharm

import pandas as pd
import os
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
import matplotlib.pyplot as plt
from pylab import mpl
from scipy.cluster import hierarchy
mpl.rcParams['font.sans-serif'] = ['KaiTi']


def cluster(t, criterion):
    news_data_files = os.listdir(r'../CompareTrail/ZhangElectric/final_input')
    path_cluster = r'../CompareTrail/ZhangElectric/zhang_cluster/{0}_cluster.txt'.format(t)
    f_cluster = open(path_cluster, 'w', encoding='utf-8')
    for news_data_file in news_data_files:
        path = r'../CompareTrail/ZhangElectric/final_input/' + news_data_file
        news_data = pd.read_csv(path, encoding='utf-8')
        terms = news_data.columns.tolist()
        rdata = news_data.values
        # print("rdata:\n", rdata)
        # 这里的linkage可以为single,complete，average，weighted，centroid等
        linkage_type = "average"
        linkage_matrix = linkage(rdata, linkage_type)
        # plt.figure(dpi=900)
        # dendrogram(linkage_matrix, labels=terms, orientation='left', leaf_font_size=3)
        # plt.title(linkage_type + "link")
        # plt.savefig('../results/cluster_jpg/' + news_data_file.split('_')[0] + '.jpg')
        labels = fcluster(linkage_matrix, t=t, criterion=criterion)
        tmp = {}

        for d, l in zip(labels, terms):
            tmp.setdefault(d, []).append(l)
        out = {}
        # path = '../results/distance_clusterV1/{0}_cluster.txt'.format(t)

        for k in sorted(tmp):
            # print(k, tmp[k])
            if len(tmp[k]) > 4:
                f_cluster.write(str(tmp[k]) + '\n')
        # f_cluster.write(path + '\n')
    f_cluster.close()
    print(t, "****************聚类完成**********")
