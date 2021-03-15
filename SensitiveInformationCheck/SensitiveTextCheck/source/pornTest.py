import requests
import base64
import json
import time
import os
import urllib3


# 构建请求串
def mak_req(img_domain='', filename='', aigroup='ai_g7', ais='keyword',
            textdata='', debug='y', loc='y', addition_info_in_place_ai='y',
            todo='', word='', kindnum=4):
    # if filename !='':
    rs = {}

    if textdata == '':
        rs.update({"imgurl": img_domain + filename})
        rs.update({"textdata": '0'})
    else:
        rs.update({"textdata": textdata})
        # rs.update({"imgurl": ''})
    if debug != '':
        rs.update({"debug": debug})
    if debug == 'y':
        rs.update({"picid": filename})
    if loc == 'y' or loc != 'n':
        rs.update({"loc": loc})
    if addition_info_in_place_ai == 'y' or addition_info_in_place_ai == 'n':
        rs.update({'addition_info_in_place_ai': addition_info_in_place_ai})
    # 请求的aigroup
    rs.update({"ais": ais})

    if aigroup == '':  # aigrou如果为空, 构造使用默认aigroup
        aigroup = self.aigroup
        rs.update({'aigroup': self.aigroup})
    elif aigroup == 'del':  # 如果aigroup 为del ，构造 请求中不加aigroup字段，应该返回错误
        print()
    elif aigroup == 'empty':  # 如果aigroup 为empty ，构造 请求中aigroup置空，应该返回错误
        rs.update({'aigroup': ''})
    else:
        rs.update({'aigroup': aigroup})

    if todo != '':
        rs.update({"todo": todo})

    if word != '':
        rs.update({"word": word})

    if kindnum != '':
        rs.update({"kindnum": kindnum})
    # print(str(rs))
    return rs


# 进行ai_group 请求，得到返回结果
def do_req(rs, ai_url):
    # do  http ai request
    http = urllib3.PoolManager()

    r = http.request(
        'post',
        ai_url,
        fields=rs
    )
    # print(r.data.decode())

    return r

if __name__=='__main__':

    # 读取敏感词文件
    sensitivityList = []
    pornCount = 0
    count = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    pornNum = 0
    #加载涉黄文本
    pornfile = open("../porn.txt",encoding='UTF-8')
    for line in  pornfile :
        pornNum = pornNum + 1
        dict1 = {}
        # dict1['type'] = 'porn'
        dict1['content'] = line
        sensitivityList.append(dict1)
    pornfile.close()

    # print(sensitivityList)
    url = 'http://localhost:5000'
    aiurl = '{}/ai'.format(url)
    i = 0
    for dicData in sensitivityList:
        textdata = dicData['content']
        rs = mak_req(filename='22123456'+str(i), textdata=textdata, aigroup='ai_g7', ais='keyword#cherry')
        i = i+1
        r = do_req(rs, aiurl)
        d = json.loads(r.data.decode())
        # print (d)
        result = d['result']
        # print(result)
        j = 0
        for resultModel in result :
            code = resultModel['code']
            if code == 1:
                mr = resultModel['mr']
                senclass = mr[0]['class']
                if senclass.find('porn')>0 :
                    print("涉黄")
                    pornCount = pornCount + 1
                    tp = tp + 1
                    break
            if j == len(result)-1:
                print("其他")
                fn = fn + 1
            j = j+1
    normalList = []
    file1 = open("../normalCheck.txt",encoding='UTF-8')
    for line1 in file1 :
        normalList.append(line1)
    file1.close()
    m = 0
    for textdata in normalList:
        rs = mak_req(filename='22123456'+str(m), textdata=textdata, aigroup='ai_g7', ais='keyword#cherry')
        i = i+1
        r = do_req(rs, aiurl)
        d = json.loads(r.data.decode())
        # print (d)
        result = d['result']
        # print(result)
        j = 0
        for resultModel in result :
            code = resultModel['code']
            if code == 1:
                mr = resultModel['mr']
                senclass = mr[0]['class']
                if senclass.find('porn')>0 :
                    print("涉黄")
                    fp = fp + 1
                    break
            if j == len(result)-1:
                print("其他")
                tn = tn + 1
            j = j+1
    print("**********************涉黄类型最终检测结果****************************")
    print("涉黄准确率为：{}".format((tp+tn)/(tp+fp+tn+fn)))
    print("涉黄召回率为：{}".format(tp/(tp+fn)))
    accuracy = tp/(tp+fp)
    recall = tp/(tp+fn)
    print("涉黄F值为：{}".format((accuracy*recall*2)/(accuracy+recall)))
