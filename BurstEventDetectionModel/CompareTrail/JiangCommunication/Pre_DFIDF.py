# _*__coding:utf-8 _*__
# @Time :2022/9/20 0020 16:09
# @Author :bay
# @File Pre_DFIDF.py
# @Software : PyCharm
import math
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from FunctionTrail.code5jieba_fenci import fenci_word
from datetime import timedelta as td
import datetime
import json


# 分词后的单词向量化
def get_transformed_dataframe(documents):
    count_vectorizer = CountVectorizer(min_df=0, max_df=1.0)
    fit_transformed_data = count_vectorizer.fit_transform(documents)
    fit_transformed_dataframe = pd.DataFrame(fit_transformed_data.A, columns=count_vectorizer.get_feature_names_out())
    return fit_transformed_dataframe


# 去除重复的title和url，并实现title分词
def _get_news_data_with_tokenized(news_dataframe):
    news_dataframe = news_dataframe.drop_duplicates(subset=['content_check']).reset_index(drop=True)
    # news_dataframe = news_dataframe.drop_duplicates(subset=['url']).reset_index(drop=True)
    # print("news_dataframe:\n", news_dataframe)
    news_dataframe['title_tokenized'] = news_dataframe.content_check.apply(lambda x: fenci_word(x))
    return news_dataframe


def get_word_dic():
    word_dics = {}
    for i in range(0, time_cha+1):
        temp_time = (start_time + td(days=i)).strftime("%Y-%m-%d")
        # print("temp_time", temp_time)
        temp_data = news_data[news_data['time'] == temp_time]
        news_dataframe = _get_news_data_with_tokenized(temp_data)
        word_list = get_transformed_dataframe(news_dataframe.title_tokenized)
        # print("单词向量化：\n", word_list)
        doc_num = word_list.shape[0]
        # 包含词的文档数 Yj,t 第t天包含j的博文
        word_dic = {}
        for word in tqdm(word_list.columns):
            word_list[word] = word_list[word].apply(lambda x: 1 if x > 0 else 0)
            word_dic[word] = word_list.iloc[:][word].sum() / doc_num
        # print("word_dic: ", word_dic)
        word_dics[temp_time] = word_dic
    file_path = r'D:\workspace\pycharm\PaperTrail\CompareTrail\JiangCommunication\word_dict_ttV2.json'
    # print(file_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(word_dics, f, ensure_ascii=False, indent=4)
    print("存储完成")
    # return word_dics


if __name__ == '__main__':
    news_data = pd.read_csv(r'D:\workspace\pycharm\PaperTrail\corpus\data20220911\test_total_data_process2.csv').astype(str)
    # TODO 1、按天计算
    # 取出当天的数据
    news_data['time'] = news_data['release_time'].apply(lambda x: str(x)[0:10])
    start_time = datetime.date(*map(int, news_data['time'].min().split('-')))
    end_time = datetime.date(*map(int, news_data['time'].max().split('-')))
    time_cha = (end_time - start_time).days
    # t = 10
    # time_diff = time_cha - t
    # print("time_diff", time_diff)
    get_word_dic()