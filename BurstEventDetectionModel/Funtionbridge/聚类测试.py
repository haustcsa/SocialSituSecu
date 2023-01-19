# @Time : 2022/6/27 0027 15:21
# @Author : BAY
# @File : 聚类测试.py
# @Software : PyCharm
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from 聚类综合测试文件 import cluster as ct
from 谱聚类方法 import Spectral_cluster
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签



# 拟合结果绘图
def scatter_plot(X, y, pred, n, C=None, outlier=False, edgecolors='black', alpha=1.0,
                 title=''):
    colors = ['red', 'blue', 'green', 'orange']
    plt.figure(figsize=(10, 4))
    # 基准结果
    plt.subplot(121)
    plt.title('target', fontsize=14)
    for i in range(n):
        plt.scatter(X[:, 0][y == i], X[:, 1][y == i], c=colors[i], alpha=alpha,
                    edgecolors=edgecolors, label='y=%d' % i)
    # plt.legend()
    # 预测结果
    plt.subplot(122)
    plt.title('predict', fontsize=14)
    for i in range(n):
        plt.scatter(X[:, 0][pred == i], X[:, 1][pred == i], c=colors[i], alpha=alpha,
                    edgecolors=edgecolors, label='p=%d' % i)
    # 簇中心
    if type(C) != type(None):
        plt.scatter(C[:, 0], C[:, 1], s=70, c='yellow', marker='v', alpha=alpha,
                    edgecolors=edgecolors, label='centers')
    if outlier == True:
        plt.scatter(X[pred == -1, 0], X[pred == -1, 1], c='yellow', alpha=alpha,
                    edgecolors=edgecolors, label='outlier')
    # plt.legend()
    # y=1.02
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.show()


# 没毛病
def Kmeans(X, terms):
    # TODO 用PCA看可视化效果
    # pca = PCA(n_components=2)
    # result = pca.fit_transform(X)
    # # 可视化显示
    # ig, ax = plt.subplots(figsize=(15, 15))
    # # TODO 需要修改
    # # s must be a scalar, or float array-like with the same size as x and y
    # # 不能是terms而是每个单词出现的频率
    # ax.scatter(result[:, 0], result[:, 1], c='SeaGreen', s=terms, alpha=0.5)  # 绘制散点图，圆圈就是出现的频率
    #
    # # 不同的圆圈有时会重叠在一起，可以使用adjustText修正文字重叠现象
    # from adjustText import adjust_text
    # new_texts = [plt.text(x, y, text, fontsize=12) for x, y, text in zip(result[:, 0], result[:, 1], words)]
    #
    # adjust_text(new_texts,
    #             only_move={'text': 'x'},
    #             arrowprops=dict(arrowstyle='-', color='grey'))
    #
    # # 美观起见隐藏顶部与右侧边框线
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.show()
    from sklearn.manifold import TSNE
    ts = TSNE(2)
    result2 = ts.fit_transform(X)

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.scatter(result2[:, 0], result2[:, 1], c='SeaGreen', s=terms, alpha=0.5)  # 绘制散点图

    # 使用adjustText修正文字重叠现象
    # 不同的圆圈有时会重叠在一起，可以使用adjustText修正文字重叠现象
    from adjustText import adjust_text
    new_texts = [plt.text(x, y, text, fontsize=12) for x, y, text in zip(result2[:, 0], result2[:, 1], words)]

    adjust_text(new_texts,
                only_move={'text': 'x'},
                arrowprops=dict(arrowstyle='-', color='grey'))
    new_texts = [plt.text(x, y, text, fontsize=12) for x, y, text in zip(result2[:, 0], result2[:, 1], words)]
    adjust_text(new_texts,
                only_move={'text': 'x'},
                arrowprops=dict(arrowstyle='-', color='grey'))

    # 美观起见隐藏顶部与右侧边框线
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()
    ss = []
    N_cluesters = 10
    for n_clusters in range(2, N_cluesters+1):
        # n_clusters = 10
        km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=1000, n_init=1, verbose=False, random_state=0)
        # # 类似数组的稀疏矩阵
        km_X_tfidf = km.fit(X)
        scores = km.score(X)
        print("scores", scores)
        # prdict = km.predict(X)
        # # 轮廓系数
        # s0 = metrics.silhouette_score(X, prdict, metric='euclidean')
        Xs = [X]
        for index, method in enumerate([km_X_tfidf]):
            X = Xs[index]
            print("Silhouette Coefficient: %0.3f"
                  % metrics.silhouette_score(X, method.labels_, metric='euclidean'))
            ss.append(metrics.silhouette_score(X, method.labels_, metric='euclidean'))
            print("----------------------------------------------------------------")

        print("km-tfidf Top terms per cluster:")
        for i in range(n_clusters):
            # print("n_clusters\n", n_clusters)
            order_centroids = km_X_tfidf.cluster_centers_.argsort()[:, ::-1]
            print("Cluster %d:" % i, end='')
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind], end='')
            print()
    # 先用轮廓系数绘制学习曲线找出最优类别数
    krange = list(range(2, N_cluesters+1))  # 看一下在2-15之间的得到
    plt.plot(krange, ss)  # 当K为3时，最接近1
    plt.show()
    plt.savefig('../results/pictures/kmeans_choose_center.jpg')


# TODO 注意：这里是标准实现，时间复杂度为O(n^3),空间复杂度为O(n^2)，
#      时间和空间开销都很大，只能在很小的数据集上运行
# 需要继续深入研究，运行没毛病
def ct_Agglomerative(X):
    # TODO 凝聚式层次聚类

    for n in range(2, 7):
        ag0 = ct.Agglomerative(clusters_n=n)
        # pred0 = ag0.fit_predict(X)
        g_h = ag0.fit_predict(X, return_all=True)
        print(g_h)
    # score0 = g_h.assess()
    # score0 = ag0.assess()
    # print("score0: ", score0)

    # scatter_plot(X, y, pred0, cn, title='[agglomerative]')


# TODO 可以正常运行，谱聚类的效果怎么评估？
#  还需深入研究
def spectral(X):
    print("谱聚类开始了")
    for n in range(2, 15):
        y_pred, clusterer = Spectral_cluster.spectral_clustering(X, n_clusters=n)
        print(y_pred, clusterer)


def ct_DBSCAN():
    # TODO DBSCAN 结果不太懂
    # 由于是基于密度的，所以在类的分界不明显时将类区分开会变得很困难
    eps, min_pts = 1, 8
    # eps, min_pts = 0.12, 4
    # eps, min_pts = 0.15, 10
    X = np.load('../corpus/Result_TFPDF.npy')
    ds0 = ct.DBSCAN(eps=eps, min_pts=min_pts)
    pred0 = ds0.fit_predict(X, divide_outlier=False)
    print(pred0)

    # 分类过少需要提高密度要求，即减少eps或增加min_pts
    # 分类过多需要降低密度要求，即增加eps或减小min_pts
    # scatter_plot(X, y, pred0, cn, outlier=True, title='[dbscan]')

    # 外在方法评估
    # score0 = ds0.assess(pred0, y)
    # print('外在方法评估 dbscan train score: %f' % score0)


if __name__ == '__main__':
    news_data = pd.read_csv('../results/final_result/2022-07-08_input_v1.csv', encoding='utf-8')
    terms = news_data.columns.tolist()
    X = news_data.values
    Kmeans(X, terms)
    # 有问题 ValueError: The condensed distance matrix must contain only finite values.
    # hierarchical(X)
    # ct_Agglomerative(X)
    # spectral(X)
    # ct_DBSCAN()







