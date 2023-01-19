# _*__coding:utf-8 _*__
# @Time :2022/10/13 0013 16:25
# @Author :bay
# @File code1constructionofmeaningfulstringdictionary.py
# @Software : PyCharm
import re
import jieba.posseg as pseg
import jieba
import pandas as pd


def fenci_word(doc):
    doc = doc.replace(" ", "")
    stop_words = []
    stop_words_dir = r'D:\workspace\pycharm\PaperTrail\FunctionTrail\baidu_stopwords.txt'
    for stop in open(stop_words_dir, encoding='utf-8'):
        stop_words.append(stop.replace('\n', ''))
    flags = ['a', 'an', 'ad', 'b', 'i', 'j', 'l', 'n', 'nr', 'nrt',
             'ns', 'nt', 'nw', 'nz', 's', 't', 'v', 'vn', 'PEG', 'LOC', 'ORG']
    user_dict_dir = r'D:\workspace\pycharm\PaperTrail\FunctionTrail\user_dict.txt'
    jieba.load_userdict(user_dict_dir)
    words = ''
    if len(doc) > 5:

        words = " ".join([k for k, flag in pseg.lcut(doc) if k not in stop_words and flag in flags
                 and len(k) > 1])
    # print("words", words)
    return words


def get_tags(contents):
    tags = []
    tag_lists = []
    for line in contents:
        tag = re.findall('#(.*?)#|【(.*?)】|《(.*?)》', line)
        # print("tag", tag)
        # [('易烊千玺个人独资企业申请注销', '', '')]
        if len(tag) > 0:
            for lists in tag:
                for list in lists:
                    # print(list)
                    if len(list) > 0:
                        tag_lists.append(list)
    tag_lists = set(tag_lists)
    # print(len(tag_lists))
    # print(tag_lists)
    for tag_list in tag_lists:
        words = fenci_word(tag_list)
        # print(words)
        words = words.split(' ')
        for word in words:
            if len(word) > 0:
                tags.append(word)
    return tags


if __name__ == '__main__':
    news_data = pd.read_csv(r'D:\workspace\pycharm\PaperTrail\corpus\data20220911\test_total_data_process2.csv').astype(str)
    news_data['time'] = news_data['release_time'].apply(lambda x: str(x)[0:10])
    start_data = news_data['time'] >= '2022-07-04'
    end_data = news_data['time'] <= '2022-07-13'
    data = news_data[start_data & end_data]
    contents = data['content_check'].values.tolist()
    get_tags(contents)
    tags = get_tags(contents)
    print(len(tags))
    # print(tags)
    tags = set(tags)
    print(len(tags))
    # print(tags)
    json_path = r'D:\workspace\pycharm\PaperTrail\CompareTrail\ZhouCIAC\tags.txt'
    f = open(json_path, 'w', encoding='utf-8')
    for tag in tags:
        f.write(tag)
        f.write('\n')
        # json.dump(new_wordDict, f, ensure_ascii=False, indent=4)
    print("tags存储完成")


