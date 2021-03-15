import requests
import base64
import json
import time
import os
import urllib3
import predictWithMyModel


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
    #************************敏感文本+敏感图片******************
    # 读取敏感词文件
    sensitivityList = []
    # 文本对应的图片
    img_path_list = []
    count = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    # 读取文本敏感
    file = open("sensitivityCheck.txt",encoding='UTF-8')
    for line in file :
        sensitivityList.append(line)
    file.close()
    #读取图片敏感文本
    files = os.listdir("data/validation/")
    for file in files:
        if os.path.isdir("data/validation/" + file):
            for chlidFile in os.listdir("data/validation/" + file):
                if not chlidFile[0] == "." or  not  chlidFile.find("netural") >0:
                    dict1 = dict()
                    dict1["type"] = file
                    dict1["filePath"] = "data/validation/" + file + "/" + chlidFile
                    img_path_list.append(dict1)
                   # if file == "political":
                    #    politicalCount = politicalCount + 1
                   # if file == "porn":
                    #    pornCount = pornCount + 1
                  #  if file == "terrorism":
                    #    terrorismCount = terrorismCount + 1
    print(img_path_list)
    print("开始预测：")
    # print(sensitivityList)
    #************************先预测文本的敏感性*************************
    url = 'http://localhost:5000'
    aiurl = '{}/ai'.format(url)
    i = 0
    for textdata in sensitivityList:
        rs = mak_req(filename='22123456'+str(i), textdata=textdata, aigroup='ai_g7', ais='keyword#cherry')
        textResult = 0
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
                textResult = 1
                if senclass.find('political')>0 :
                    print("敏感文本被检测为涉政")
                    break
                if senclass.find('porn')>0 :
                    print("敏感文本被检测为涉黄")
                    break
                if senclass.find('terrorism')>0 :
                    print("敏感文本被检测为文本涉恐")
                    break
            if j == len(result)-1:
                print("敏感文本被检测为正常")
                textResult = 0
            j = j+1
        #*********************检测文本对应的图片敏感性*********************
        if i<=len(img_path_list):
            predictClass, predictAccuracy  = predictWithMyModel.predictWithImagePath(img_path_list[i-1])
            if (predictAccuracy * 0.5 + textResult *0.5) > 0.5 :
                 tp = tp + 1
                 print('融合后的图文被检测为敏感')
            else:
                fn = fn + 1
        else:
            if textResult >0 :
                tp = tp + 1
            else:
                fn = fn + 1
   #***********************检测正常文本和正常图片*******************************
    #加载正常的文本
    normalList = []
    file1 = open("normalCheck.txt",encoding='UTF-8')
    for line1 in file1 :
        normalList.append(line1)
    file1.close()

    #加载正常图片
    normalImageList = []
    for imageFile in os.listdir("data/NormalImage/netural/") :
        dict1 = dict()
        dict1["type"] = "normal"
        dict1["filePath"] = "data/NormalImage/netural/" + imageFile
        normalImageList.append(dict1)
    m = 0
    for textdata in normalList:
        rs = mak_req(filename='22123456'+str(m), textdata=textdata, aigroup='ai_g7', ais='keyword#cherry')
        m = m+1
        textResult1 = 0
        r = do_req(rs, aiurl)
        d = json.loads(r.data.decode())
        # print (d)
        result = d['result']
        # print(result)
        j = 0
        for resultModel in result :
            code = resultModel['code']
            if code == 1:
                textResult1 =0
                mr = resultModel['mr']
                senclass = mr[0]['class']
                if senclass.find('political')>0 :
                    print("正常文本被检测为涉政")
                    break
                if senclass.find('porn')>0 :
                    print("正常文本被检测为涉黄")
                    break
                if senclass.find('terrorism')>0 :
                    print("正常文本被检测为涉恐")
                    break
            if j == len(result)-1:
                textResult1 = 1
                print("正常文本被检测为正常")
            j = j+1
        #*********************检测文本对应的图片敏感性*********************
        if m<=len(normalImageList):
            predictClass, predictAccuracy  = predictWithMyModel.predictWithImagePath(normalImageList[m-1])
            if ((1-predictAccuracy) * 0.5 + textResult1 *0.5) > 0.5 :
                 tn = tn + 1
                 print('正常图文在融合后被检测为非敏感')
            else:
                fp = fp + 1
        else:
            if textResult1 >0 :
                tn = tn + 1
            else:
                fp = fp + 1
    print("准确率为：{}".format((tp+tn)/(tp+fp+tn+fn)))
    print("召回率为：{}".format(tp/(tp+fn)))
    accuracy = tp/(tp+fp)
    recall = tp/(tp+fn)
    print("F值为：{}".format((accuracy*recall*2)/(accuracy+recall)))
