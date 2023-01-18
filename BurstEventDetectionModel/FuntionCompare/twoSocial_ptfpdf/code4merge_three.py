# -*--coding:utf-8 -*--
# @Time : 2022/7/27 0027 16:04
# @Author : BAY
# @File : select_bursty_value.py
# @Software : PyCharm

import os
import json


def read_files():
    # 读取词频增长率文件夹中的文件
    # increment_files = os.listdir()
    # # TODO 果真不按顺序进行输出，所以。。。
    # # increment_files.sort(key=lambda x: int(x[:x.find("_")]))  # 按照前面的数字字符排序
    # print("再次验证文件是不是按顺序的", len(increment_files), increment_files)
    tfpdf_files = os.listdir(r'D:\workspace\pycharm\PaperTrail\results\word_TFPDF')
    # TODO 对比实验
    # tfpdf_files = os.listdir(r'D:\workspace\pycharm\PaperTrail\CompareTrail\ZhangElectric\TFIDF_word')
    # print("验证文件是不是按顺序的", len(tfpdf_files), tfpdf_files)
    social_files = os.listdir(r'D:\workspace\pycharm\PaperTrail\results\word_social')
    # print("验证文件是不是按顺序的", len(social_files), social_files)
    # social_files.sort(key=lambda x: int(x[:x.find("_")]))
    # # print("验证文件是不是按顺序的", len(social_files), social_files)
    # increment_data = []
    # for increment_file in increment_files:
    #     f = open(r'D:\workspace\pycharm\PaperTrail\results\word_TFPDF' + '\\' + increment_file, 'r', encoding='utf-8')
    #     data = json.load(f)
    #     increment_data.append(data)
    tfpdf_data = []
    for tfpdf_file in tfpdf_files:
        f = open(r'D:\workspace\pycharm\PaperTrail\results\word_TFPDF' + '\\' + tfpdf_file, 'r', encoding='utf-8')
        data = json.load(f)
        tfpdf_data.append(data)
    social_data = []
    for social_file in social_files:
        f = open(r'D:\workspace\pycharm\PaperTrail\results\word_social' + '\\' + social_file, 'r', encoding='utf-8')
        data = json.load(f)
        social_data.append(data)

    for i_data, j_data in zip(tfpdf_data, social_data):
        for key in i_data.keys():
            if key in j_data.keys():
                i_data[key] = i_data[key] * j_data[key]
            else:
                i_data[key] = i_data[key]
    return tfpdf_data


# 计算突发度
def compute_butsty(tfpdf_data, V):
    print(len(tfpdf_data))
    N = 10
    new_lists = []
    new_a_list = {}
    for i in range(N, N+10):
        T1 = tfpdf_data[i]
        # print("最初的值", T1)
        T1_update = dict.fromkeys(T1.keys(), 0)
        # print("所有值赋为0", T1_update)
        for key1 in T1_update.keys():
            # print("key1:", key1)
            for j in range(i-N, i):
                if key1 in tfpdf_data[j].keys():
                    T1_update[key1] += T1[key1] - tfpdf_data[j][key1]
            T1_update[key1] = round(T1_update[key1] / N, 7)
            # print("计算的结果", T1[key1])
        for key1 in T1_update.keys():
            if T1_update[key1] == 0:
                T1_update[key1] = round(T1[key1]/N, 7)
        # 排序
        T1_update = dict(sorted(T1_update.items(), key=lambda x: x[1], reverse=True))
        new_a = {}
        for i, (k, v) in enumerate(T1_update.items()):
            if v >= 0.0007138:
                new_a[k] = v
        # # TODO 突发度的判断
        new_a_list.update(new_a)
        new_list = list(new_a)
        new_lists.append(new_list)
    # print("111 new_a_list", new_a_list)
    new_a_list = dict(sorted(new_a_list.items(), key=lambda x: x[1], reverse=True))
    print("222 new_a_list", len(new_a_list), new_a_list)
    file_path = './Wij/bursty_value_v{0}.json'.format(V)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(new_a_list, f, ensure_ascii=False, indent=4)
    print("存储完成")
    path = './word_zmerge/bursty_v{0}.txt'.format(V)
    # path = r'D:\workspace\pycharm\PaperTrail\CompareTrail\ZhangElectric\TOP100\bursty_v{}.txt'.format(V)
    # path = r'D:\workspace\pycharm\PaperTrail\CompareTrail\_WangComputer\word_zmerge\bursty_v2.txt'
    with open(path, 'w', encoding='utf-8') as f:
        for key in new_lists:
            f.write(" ".join(key))
            f.write('\n')
    f.close()

    # print("T{0}天的词语突发度".format(i-N), T1_update)


def merge_three(V):
    # 测试数据
    # tfpdf_data = [
    #     {"病例": 1.0,"感染者": 0.8834951,"确诊": 0.8543689,"上海": 0.815534,"新增": 0.7087379,},
    #     {"病例": 1.0, "确诊": 0.949607,"乌克兰": 0.5545076, "核酸检测": 0.5254739,"居住": 0.4952381,},
    #     {"病例": 1.0, "确诊": 0.9324046,"核酸检测": 0.6436969, "诊断": 0.4863026,"企业": 0.4404079,},
    #     {"病例": 1.0,"宇宙": 0.9339154,"确诊": 0.9141044,"美国": 0.8664662, "斯特兰": 0.8493151,},
    #     {"农民工": 1.0,"县城": 0.9240506, "肖战": 0.8661368,"工作": 0.8342691, "人口": 0.6788943,},
    #     {"孩子": 1.0, "信息": 0.7474214, "情况": 0.7381527,"农民工": 0.6817972,"学生": 0.6718716,},
    #     {"秦怡": 1.0, "银行": 0.8820681,"居住": 0.6984127, "新增": 0.669739, "病例": 0.6659564,},
    #     {"俄罗斯": 1.0,  "新增": 0.7543781, "疫情": 0.5963255, "专业": 0.5849867, "感染者": 0.5287433,},
    #     {"美国": 1.0,"工作": 0.8604247, "病例": 0.8486188,"新增": 0.7718119, "台湾": 0.7593241,},
    #     {"美国": 1.0, "人员": 0.5467499, "死亡": 0.4930708, "工作": 0.3851063, "疫情": 0.3508422,},
    #     {"病例": 1.0, "确诊": 0.7995256, "偶像": 0.6717891, "资本": 0.6414348,"美国": 0.6325192,"bay":45},
    #
    #   ]
    # read_files()
    tfpdf_data = read_files()
    compute_butsty(tfpdf_data, V)


if __name__ == '__main__':
    merge_three(2)









