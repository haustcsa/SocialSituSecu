# _*__coding:utf-8 _*__
# @Time :2022/10/18 0018 10:58
# @Author :bay
# @File code3BurstyScore.py
# @Software : PyCharm
import os
import json


def read_file():
    Bursty_files = os.listdir(r'D:\workspace\pycharm\PaperTrail\CompareTrail\ZhouCIAC\BurstyTermScore')
    Bursty_data = []
    for bursty_file in Bursty_files:
        f = open(r'D:\workspace\pycharm\PaperTrail\CompareTrail\ZhouCIAC\BurstyTermScore' + '\\' + bursty_file, 'r', encoding='utf-8')
        data = json.load(f)
        Bursty_data.append(data)
    return Bursty_data


def compute_butsty(Bursty_data, V):
    print(len(Bursty_data))
    new_lists = []
    new_a_list = {}
    for i in range(10):
        T1 = Bursty_data[i]
        # print("最初的值", T1)
        # 排序
        T1_update = dict(sorted(T1.items(), key=lambda x: x[1], reverse=True))
        # 每天的关键词
        # print("这个T1_update是个啥", T1_update)
        new_a = {}
        for i, (k, v) in enumerate(T1_update.items()):
            if v >= 0.0006479:
                new_a[k] = v
        # TODO 突发度的判断
        new_a_list.update(new_a)
        new_list = list(new_a)
        print("new_list", new_list)
        new_lists.append(new_list)
    # new_list = list(new_key_list)
    # print("new_lists", new_lists)
    new_a_list = dict(sorted(new_a_list.items(), key=lambda x: x[1], reverse=True))
    print(len(new_a_list))
    file_path = r'D:\workspace\pycharm\PaperTrail\CompareTrail\ZhouCIAC\Fwi\bursty_value_v{0}.json'.format(V)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(new_a_list, f, ensure_ascii=False, indent=4)
    print("存储完成")
    path = r'D:\workspace\pycharm\PaperTrail\CompareTrail\ZhouCIAC\word_zmerge\bursty_v{0}.txt'.format(V)
    with open(path, 'w', encoding='utf-8') as f:
        for key in new_lists:
            f.write(" ".join(key))
            f.write('\n')
    f.close()


if __name__ == '__main__':
    V = 3
    Bursty_data = read_file()
    compute_butsty(Bursty_data, V)



