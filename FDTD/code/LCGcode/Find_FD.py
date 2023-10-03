# -*- coding:utf-8 -*-
# @Time      :2023/5/11 16:04
# @FileName  :Find_FD.py
# @Author    :GDJ
# @Software  :PyCharm

import pandas as pd
import warnings
import time

t0 = time.time()

warnings.filterwarnings('ignore')
from itertools import combinations

data = pd.read_csv('../data/restaurant/cmp_result.csv', sep=',', header=None,
                   names=['name', 'BUILDING', 'STREET', 'ZIPCODE', 'PHONE', 'ID'])

# data = pd.read_csv('../../data/restaurant/restaurant_truth.csv', sep=',', header=None)
# data.columns = list('ABCDEF')
# 删除指定列(编号列)
data = data.drop(columns=["ID"])


def Functional_dependence(LHS, RHS):
    x = data[LHS].values.tolist()
    y = data[RHS].values.tolist()
    l = len(x)
    num = 0
    for i in range(l):
        if sum(x[i]) == 0:
            if y[i] != 0:
                return False, num
            else:
                num += 1
    return True, num


def fff(LHS, RHS):
    lhs = LHS.copy()
    LHS.append(RHS)
    # print(list(data[LHS].groupby(lhs)))
    # print('-' * 30)


def compute_entropy(LHS, RHS):
    # unique()方法返回的是去重之后的不同值，而nunique()方法则直接放回不同值的个数
    tmp = data.groupby(LHS)[RHS].nunique()
    entropy = (tmp > 1).sum()
    return entropy


def findFD():
    FD_list = []
    # print('data.shape:', data.shape)
    correlation = data.corr(method='spearman')
    # print(type(correlation), correlation)
    # data.shape[1]列数
    for r in range(2, data.shape[1]):
        # print('r:', r, '-' * 30)
        # 创建一个迭代器，返回iterable中所有长度为r的子序列，返回的子序列中的项按输入iterable中的顺序排序。
        for comb in combinations(data.columns, r):
            # print('comb:', comb)
            for RHS in comb:
                # print('RHS:', RHS, end='\t')
                LHS = [col for col in comb if col != RHS]
                # fff(LHS, RHS)
                flag, num = Functional_dependence(LHS, RHS)
                if flag:
                    for i in LHS:
                        if correlation[i][RHS] == 0:
                            break
                    FD_list.append([LHS, RHS, num])
                    # print('LHS,RHS:', LHS, RHS)
    # print(len(FD_list))
    FD_list = sorted(FD_list, key=lambda x: x[2], reverse=True)
    # print(FD_list)
    # print("\t=> Execution Time: {} seconds".format(time.time() - t0))
    dc = '1.0:not('
    for i in FD_list[0][0]:
        dc += 't1.%s=t2.%s&' % (str(i), str(i))
    print('1.0:not(t1.BUILDING=t2.BUILDING&t1.STREET=t2.STREET&t1.ZIPCODE!=t2.ZIPCODE)')
    dc += 't1.%s!=t2.%s)' % (FD_list[0][1], FD_list[0][1])
    # print(dc)
    return [dc]


if __name__ == '__main__':
    # print('ok')
    findFD()
