## 增加数据
## 微博数据爬虫.py 按话题和时间去爬取数据


## 1. 执行0.get_SqlData.py文件，从数据库中获取原始数据（需要的时间段）
## 2. 执行0.process_data.py文件，对原始数据进行处理
    ## 先执行process_data()加列名并处理数据来源
    ## 接着执行danwei()将点赞、转发、评论含有万的转化为0000
## 3 统计每天的数据分布
    ## 作为论文中的插图
## 3 按天统计词频并画出最高词频的图
    ## 作为论文中额插图

## 1. code0SocialMediaInfluenceV2.py 文件，获取社交媒体影响力
    ## 得到三个值,其中百度贴吧没有点赞数、转发数、评论数
    ##（还得在考虑,目前已考虑好引入了Alexa排名相关参数）
## 进而 code1SocialMediaInfluenceV2.py 文件，获取每个词的社交媒体影响力权重
    ## 保存在result文件夹中

## 2. 计算词频增长率 执行code2freqincrement.py文件
    ## 结果保存在result文件夹中
## 3.计算TFPDF的值 执行 code3TFPDFtrain.py文件
    ## 结果保存在result文件夹中
## 4. 融合三个特征 执行code4merge_three.py文件
    ## 从中选取前100个突发词

## 5code5 jieba_fenci.py文件，按天对微博进行分词并保存
 这一步是为了计算突发词相似度和突发词做对比使用
## 5. code5bursty_similarity.py文件
    ## 得到一个突发词共现度矩阵

## 6. 层次聚类函数 code6cluster.py
    ## 得到分类结果

## 7.执行main_full.py


