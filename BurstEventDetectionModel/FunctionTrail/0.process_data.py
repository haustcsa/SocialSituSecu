# _*__coding:utf-8 _*__
# @Time :2022/5/11 0011 16:36
# @Author :bay
# @File 0.process_data.py
# @Software : PyCharm
import random
import pandas as pd


# TODO 随机造三列数据并存储到csv文件中
# TODO 目前不需要了，这个函数用不到了，数据库中有这些数据
# def read_csv():
#     S_data = pd.read_csv(r'D:\workspace\pycharm\Paper_Experiment\corpus\DFIDFdata\total_data_df_idf.csv', encoding='gbk',
#                          names=['id', 'release_time', 'release_source', 'content_check'])
#     # 添加列
#     s_shape = S_data.shape[0]
#     print(S_data.shape)
#     print(s_shape)
#     Rewards = [random.randint(100, 1000) for j in range(s_shape)]
#     Comments = [random.randint(100, 1000) for j in range(s_shape)]
#     Likes = [random.randint(100, 1000) for j in range(s_shape)]
#     S_data['Rewards'] = Rewards
#     S_data['Comments'] = Comments
#     S_data['Likes'] = Likes
#     S_data.to_csv(r'D:\workspace\pycharm\Paper_Experiment\corpus\DFIDFdata\temp_total_data_df_idf.csv', index=False)


# TODO 对CSV文件中的数据进行处理 将微博热榜改为1，知乎热榜改为0.7 百度贴吧改为0.7 并保存到CSV文件中
def process_data():
    temp_data = pd.read_csv(r'../corpus/data20220911/test_total_data.csv', encoding='utf-8', header=None,
                            names=['id', 'release_time', 'release_source', 'content_check', 'rewards', 'comments', 'likes'])
    # print(temp_data.sample(5))
    # print(temp_data['release_source'].sample(5))
    temp_data.loc[temp_data['release_source'] == '微博热榜', 'release_source_code'] = 1
    temp_data.loc[temp_data['release_source'] == '知乎热榜', 'release_source_code'] = 2
    temp_data.loc[temp_data['release_source'] == '百度贴吧', 'release_source_code'] = 3
    # print(temp_data.sample(5))
    # df['价格']=df['价格'].apply(lambda x:  float(x.split('万')[0])*10000 if x.split('万')[-1]=='' else float(x)).astype(int)
    temp_data.to_csv(r'../corpus/data20220911/test_total_data_process1.csv', index=False)


# 处理10万+
def danwei():
    df = pd.read_csv(r'../corpus/data20220911/test_total_data_process1.csv', encoding='utf-8')
    df['rewards'] = df['rewards'].apply(lambda x:  float(str(x).split('万+')[0])*10000 if str(x).split('万+')[-1] == '' else float(x))
    df['comments'] = df['comments'].apply(lambda x:  float(str(x).split('万+')[0])*10000 if str(x).split('万+')[-1] == '' else float(x))
    df['likes'] = df['likes'].apply(lambda x:  float(str(x).split('万+')[0])*10000 if str(x).split('万+')[-1] == '' else float(x))
    df.to_csv(r'../corpus/data20220911/test_total_data_process2.csv', index=False)
    print(df.shape)
    # print(df['rewards'])


if __name__ == '__main__':
    # process_data()
    danwei()


