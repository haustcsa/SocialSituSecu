# -*--coding:utf-8 -*--
# @Time : 2022/8/29 0029 11:40
# @Author : BAY
# @File : 比较两个stopwords文件.py
# @Software : PyCharm

import sys
import difflib


# 读取配置文件函数
def read_file(file_name):
    try:
        file_handle = open(file_name, 'r', encoding='utf-8')
        text = file_handle.read().splitlines()  # 读取后以行进行分割
        file_handle.close()
        return text
    except IOError as error:
        print('Read file Error: {0}'.format(error))
        sys.exit()


# 比较两个文件并输出html格式的结果
def compare_file(file1_name, file2_name):
    if file1_name == "" or file2_name == "":
        print('文件路径不能为空：file1_name的路径为：{0}, file2_name的路径为：{1} .'.format(file1_name, file2_name))
        sys.exit()
    text1_lines = read_file(file1_name)
    text2_lines = read_file(file2_name)
    # TODO 方法1 + -
    d = difflib.Differ()  # 创建Differ对象
    diff = d.compare(text1_lines, text2_lines)  # 采用compare方法对字符串进行比较
    print(diff)
    # print('\n'.join(list(diff)))
    # TODO 方法2
    # diff = difflib.ndiff(text1_lines, text2_lines)
    # print("diff", diff)
    # for i in diff:
    #     print("i", i)
        # if i.startswith('+'):
        #     print(i)

    # or
    # print(set(text2_lines) - set(text1_lines))
    list = set(text2_lines) - set(text1_lines)
    for j in list:
        print(j)
    # diff = difflib.HtmlDiff()  # 创建htmldiff 对象
    # result = diff.make_file(text1_lines, text2_lines)  # 通过make_file 方法输出 html 格式的对比结果
    # #  将结果保存到result.html文件中并打开
    # try:
    #     with open('result.html', 'w') as result_file:  # 同 f = open('result.html', 'w') 打开或创建一个result.html文件
    #         result_file.write(result)  # 同 f.write(result)
    # except IOError as error:
    #     print('写入html文件错误：{0}'.format(error))


if __name__ == "__main__":
    compare_file(
        r'D:\workspace\pycharm\PaperTrail\backup\usedwords\baidu_stopwordsV1.txt',
        r'D:\workspace\pycharm\PaperTrail\FunctionTrail\baidu_stopwords.txt',)  # 传入两文件的路径

