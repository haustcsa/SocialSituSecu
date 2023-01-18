# _*__coding:utf-8 _*__
# @Time :2022/9/28 0028 19:53
# @Author :bay
# @File 停词测试.py
# @Software : PyCharm
import jieba.posseg as pseg
sentence = '我打算n年之后回家'
stop_words = ['n年']
flags = ['a', 'an', 'ad', 'b', 'i', 'j', 'l', 'n', 'nr', 'nrt',
             'ns', 'nt', 'nw', 'nz', 's', 't', 'v', 'vn', 'PEG', 'LOC', 'ORG']
result = []
for k, flag in pseg.lcut(sentence):
    if flag in flags:
        if k not in stop_words:
            result.append(k)
print(result)

