# _*__coding:utf-8 _*__
# @Time :2022/10/17 0017 15:34
# @Author :bay
# @File 字典相加测试.py
# @Software : PyCharm
import pandas as pd

import numpy as np
def sum_dict(a, b):
    temp = dict()
    # python3,dict_keys类似set； | 并集
    for key in a.keys() | b.keys():
        temp[key] = np.mean([d.get(key, 0) for d in (a, b)])
    return temp


def test():
    # python3使用reduce需要先导入
    from functools import reduce
    # [a,b,c]列表中的参数可以2个也可以多个，自己尝试。
    result = (reduce(sum_dict, [a, b, c]))
    return result

# {'f': 1.6666667, 'a': 4.0, 'c': 1.0, 'b': 1.6666667, 'd': 1.3333333, 'g': 1.0}
# {'g': 1.5, 'a': 5.5, 'c': 0.75, 'b': 1.25, 'd': 1.0, 'f': 2.5}
a = {'a': 1, 'b': 2, 'c': 3}
b = {'a': 1, 'b': 3, 'd': 4}
c = {'g': 3, 'f': 5, 'a': 10}
# result = test()
# for key in result.keys():
#     result[key] = round(result[key] / 3, 7)
# print(result)
df = pd.DataFrame([a, b, c])
print(df)
df = df.fillna(0)
answer = dict(df.mean())
print(answer)
