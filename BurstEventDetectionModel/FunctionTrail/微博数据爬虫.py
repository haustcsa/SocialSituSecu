# -*--coding:utf-8 -*--
# @Time : 2022/9/4 0004 21:50
# @Author : BAY
# @File : 微博数据爬虫.py

import json
import requests
from lxml import etree
import time
import re
import pyodbc
from urllib.parse import quote
j = 0


def get_contents(data):
    # print("data:\n", data)
    content_data = []
    try:
        head_data = data['card_group'][0]['mblog']
        fabu_time = head_data['created_at']
        # 时间格式转换
        # Tue Apr 20 19:34:58 +0800 2021
        release_time1 = fabu_time[4:-11]
        release_time2 = fabu_time[-4:]
        created_at = release_time1 + ' ' + release_time2
        created_at = int(time.mktime(time.strptime(created_at, "%b %d %H:%M:%S %Y")))
        timeArray = time.localtime(created_at)  # float变为时间戳
        release_time = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)  # 时间戳转成Y-M-D的str
        # print(release_time)
        # print(type(release_time[:10]), release_time[:10])
        # times = ['2022-06-23', '2022-06-24', '2022-06-25', '2022-06-26', '2022-06-27',
        #          '2022-06-28', '2022-06-29', '2022-06-30', '2022-07-01', '2022-07-02']
        times = ['2022-07-03', '2022-07-04', '2022-07-05', '2022-07-06', '2022-07-07',
                 '2022-07-08', '2022-07-09', '2022-07-10', '2022-07-11', '2022-07-12',
                 '2022-07-13']
        if release_time[:10] in times:
            # print("pass")
            text = head_data['text']
            tree = etree.HTML(text)
            link = "https://m.weibo.cn/detail/" + head_data['id']
            # print(link)
            # link = tree.xpath('//a/@href')[0]
            content = tree.xpath('//text()')
            contents = ''.join(content)

            release_source = '微博热榜'

            likes = head_data['attitudes_count']
            rewards = head_data['reposts_count']
            commnents = head_data['comments_count']

            content_data.append(release_time)
            content_data.append(release_source)
            content_data.append(link)

            content_data.append(contents)
            content_data.append(rewards)
            content_data.append(commnents)
            content_data.append(likes)
            # print(content_data)
            save_data(content_data)

    except:
        pass
        # print("没有mblog")


def get_data(url):
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36',
    }
    response = requests.get(url, headers=headers)
    json_dict = response.json()
    # print(type(json_dict))
    # print(len(json_dict))
    if len(json_dict) == 2:
    # json_dict = json.load(response.json())
    # print(json_dict)
    # print(type(json_dict))
    # tree = etree.HTML(response.text)
        datas = json_dict['data']['cards']
        for data in datas:
            time.sleep(2)
            get_contents(data)
    else:
        pass
        # print("没有内容")


def save_data(content_data):
    global j
    # cnx = pyodbc.connect('DRIVER={SQL Server};SERVER=LAPTOP-1LD4FSHS;DATABASE=cyvodDBTest;UID=sa;PWD=123456;charset=gbk')
    cnx = pyodbc.connect('DRIVER={SQL Server};SERVER=101.201.252.178,9998;DATABASE=cyvodDB;UID=sa;PWD=Websoft9Com;charset=gbk')
    cur = cnx.cursor()
    sql_link = "SELECT 1 FROM Image_Text_Check WHERE original_link like '{}'".format(content_data[2])
    cur.execute(sql_link)
    if  cur.fetchall() != []:
        pass
        # print("文章已经存在，不执行插入")
    else:
    # class Sql:
    #     @classmethod
    #     def Society_insert_detail(cls, release_source, release_time, image_url, image_name, forwarding_number, comments,
    #                               likes, content, link):
    # 插入报道文章的详情内容
        sql_insert = "insert into Image_Text_Check( release_time, release_source,original_link, content_check, forwarding_number, comment_number, like_number)values" \
              " ('%s', '%s', '%s', '%s', '%s', '%s', '%s')" % (content_data[0], content_data[1], content_data[2], content_data[3], content_data[4], content_data[5], content_data[6])
        cur.execute(sql_insert)
        cnx.commit()

        j = j + 1
        # print("成功插入一条数据")
    cur.close()

# https://m.weibo.cn/api/container/getIndex?containerid=231522type%3D1%26t%3D10%26q%3D%23%E9%93%B6%E8%A1%8C%E8%A1%8C%E9%95%BF%E4%B8%BA%E5%A5%B3%E5%84%BF%E5%AD%A6%E4%B8%9A%E8%A2%AB%E9%AA%973.5%E4%BA%BF%23&extparam=%23%E9%93%B6%E8%A1%8C%E8%A1%8C%E9%95%BF%E4%B8%BA%E5%A5%B3%E5%84%BF%E5%AD%A6%E4%B8%9A%E8%A2%AB%E9%AA%973.5%E4%BA%BF%23&luicode=10000011&lfid=100103type%3D38%26q%3D%E9%93%B6%E8%A1%8C%E8%A1%8C%E9%95%BF%E8%A2%AB%E9%AA%973.5%E4%BA%BF%26t%3D&page_type=searchall


if __name__ == '__main__':
    #
    p_texts = [
               '刘德华谢霆锋陈伟霆唱中国人',
               '欧阳娜娜泼宋丹丹细节满分',
               '发现雪糕不标价可立即投诉',
               '雪莲5毛一包的定价13年没涨'
               ]
    for p_text in p_texts:
        for i in range(0, 30):
            params = {
                'containerid': '231522type=1&q=#{}#'.format(p_text),
                'page': i
            }
            url = 'https://m.weibo.cn/api/container/getIndex?containerid={}&page_type=searchall&page={}'.format(quote(params['containerid'], encoding='utf-8'), params['page'])

            print(url)
            # url = 'https://m.weibo.cn/api/container/getIndex?containerid=231522type%3D1%26q%3D%23%E6%96%AF%E9%87%8C%E5%85%B0%E5%8D%A1%E6%80%BB%E7%90%86%E5%AE%A3%E5%B8%83%E5%AE%9E%E6%96%BD%E5%9B%BD%E5%AE%B6%E7%B4%A7%E6%80%A5%E7%8A%B6%E6%80%81%23&page_type=searchall&page={}'.format(i)
            # https://m.weibo.cn/api/container/getIndex?containerid=231522type%3D1%26q%3D%23%E6%96%AF%E9%87%8C%E5%85%B0%E5%8D%A1%E5%AE%9E%E6%96%BD%E7%B4%A7%E6%80%A5%E7%8A%B6%E6%80%81%23&page_type=searchall
            get_data(url)
        print("j:", j)





