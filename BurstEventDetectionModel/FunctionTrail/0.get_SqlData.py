# _*__coding:utf-8 _*__
# @Time :2022/2/24 11:07
# @Author :bay
# @File 0.get_SqlData.py
# @Software : PyCharm
# TODO 需求从数据库中获取数据并保存到corpus文件夹中
import pyodbc
import csv


# 从数据库中表中查询数据
def Read_Sqldata(sql):
    # 正式库
    conn = pyodbc.connect(
        'DRIVER={SQL Server};SERVER=101.201.252.178,9998;DATABASE=cyvodDB;UID=sa;PWD=Websoft9Com;charset=utf8')
    cursor = conn.cursor()
    cursor.execute(sql)
    rs = cursor.fetchall()
    cursor.close()
    conn.close()
    return rs


def save_csv(rs, filename):
    f = open(r'D:\workspace\pycharm\PaperTrail\corpus\data7days\{}.csv'.format(filename), 'w+', newline='', encoding='utf-8')
    writer = csv.writer(f)
    for row in rs:
        writer.writerow(row)
    f.close()
    print("数据存储到{}.csv文件中".format(filename))
    print("**************Successful****************")


def Read_Save():
    sql_dicts = {
        # elect * from tb_FakeInfo
        # where datediff(day,cast('2020-03-01' as datetime),Fabu_Time) >0
        #  and (Fake_Url not like '%https://www.sohu.com/%') and ( Fake_Url not like '%http://www.sohu.com/%' )
        # 'micro_blog': "select release_time, content_check from Image_Text_Check where release_source = '微博热榜' and datediff(d,release_time,getdate()) <=24  order by release_time desc",
        # 'know': "select release_time, content_check from Image_Text_Check where release_source = '知乎热榜' and datediff(d,release_time,getdate()) <=24  order by release_time desc",
        # 'baidu_post_bar': "select release_time, content_check from Image_Text_Check where release_source = '百度贴吧' and datediff(d,release_time,getdate()) <=24  order by release_time desc", datediff(day,cast('2020-03-01' as datetime)
        'test_total_data': "select id, release_time, release_source, content_check,forwarding_number,comment_number,like_number "
                            "from Image_Text_Check "
                            "where  release_source in( '微博热榜', '知乎热榜','百度贴吧') and "
                            "datediff(day,cast('2022-06-28' as datetime),release_time) >=0  and "
                            "datediff(day,cast('2022-07-12' as datetime),release_time) <=0 "
                            "order by release_time desc"

        # 'total_data_df_idf': "select id, release_time, release_source, content_check from Image_Text_Check where  release_source in( '微博热榜', '知乎热榜','百度贴吧') and datediff(day,cast('2020-05-01' as datetime)>0  order by release_time desc"
    }
    for key, value in sql_dicts.items():
        # print(key, value)
        rs = Read_Sqldata(value)
        save_csv(rs, key)


if __name__ == '__main__':
    # 测试
    # TODO 测试后无误
    Read_Save()