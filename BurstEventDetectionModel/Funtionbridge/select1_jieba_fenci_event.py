# _*__coding:utf-8 _*__
# @Time :2022/10/5 0005 10:56
# @Author :bay
# @File select1_jieba_fenci_event.py
# @Software : PyCharm
import jieba.posseg as pseg
import jieba
import datetime
from datetime import timedelta as td


def fenci_word(doc):
    doc = doc.replace(" ", "")
    stop_words = []
    stop_words_dir = r'../FunctionTrail/baidu_stopwords.txt'
    for stop in open(stop_words_dir, encoding='utf-8'):
        stop_words.append(stop.replace('\n', ''))
    flags = ['a', 'an', 'ad', 'b', 'i', 'j', 'l', 'n', 'nr', 'nrt',
             'ns', 'nt', 'nw', 'nz', 's', 't', 'v', 'vn', 'PEG', 'LOC', 'ORG']
    user_dict_dir = r'../FunctionTrail/user_dict.txt'
    jieba.load_userdict(user_dict_dir)
    words = ''
    if len(doc) > 5:

        words = " ".join([k for k, flag in pseg.lcut(doc) if k not in stop_words and flag in flags
                 and len(k) > 1])
    # print("words", words)
    return words


if __name__ == '__main__':
    datas = open('../corpus/data20220911/event25.txt', encoding='utf-8').readlines()
    result = []
    for line in datas:
        line = line.strip()
        words = fenci_word(line)
        if len(words) > 0:
            result.append(words)
    print(result)
    new_result = []
    for i_data in result:
        j_data = i_data.split(' ')
        # print(j_data)
        for j in j_data:
            new_result.append(j)
    # print(len(new_result), '\n', new_result)
    new_result = list(set(new_result))
    print(len(new_result), '\n', new_result)
    file_path = '../results/event_values/4_event.txt'
    with open(file_path, 'w', encoding='utf-8') as f:
        for word in new_result:
            f.write(word)
            f.write('\n')
    print("存储完成")