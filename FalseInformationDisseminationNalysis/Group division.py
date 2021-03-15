import os
import jieba
import jieba.analyse
from gensim import corpora,models
import logging
def readfiles():
    path = ""
    files = os.listdir(path)
    txts_conserve = []
    file_names = []
    for file in files:
        position = path+'\\' + file
        file_names.append(file)
        with open(position, "r", encoding='gbk') as f:
            data = f.read()
            txts_conserve.append(data)
    print("加载文件")
    return txts_conserve, file_names
def process_files():
    row_txts, txts_names = readfiles()
    processed_files = []
    with open('', 'r', encoding="utf-8") as f:
        lines = f.readlines()
        f.close()
    stopwords = []
    for l in lines:
        stopwords.append(l.strip())
     def seg_sentence(sentence):
        row_sentence = sentence.replace('，', '').replace('、', '').replace('？', ''). \
            replace('//', '').replace('/', '').replace('NULL', '').lstrip()
        sentence_seged = jieba.cut(row_sentence)
        outstr = ''
        for word in sentence_seged:
            if word not in stopwords:
                if len(word) <= m1 and len(word) >=m2:
                    outstr += word
                    outstr += " "
        return outstr
    for i in range(len(row_txts)):
            line_segs = seg_sentence(row_txts[i])
            result = ''.join(line_segs)
            processed_files.append(result)
    return processed_files, txts_names
def conserve_txts():
    keywords_list = []
    user_ids = []
    processed_files, txts_names = process_files()
    for i in range(len(txts_names)):
        txt = txts_names[i].split('.')
        user_ids.append(txt[0])
    for num in range(len(processed_files)):
        finished_files = ''.join(str(i) for i in processed_files[num])
        keywords = jieba.analyse.extract_tags(finished_files, topK=k1, withWeight=False, allowPOS=())
        keywords_list.append(keywords)
        print(keywords)
    return keywords_list, user_ids
def predict_model():
    keywords_lists, user_ids = conserve_txts()
    dictionary = corpora.Dictionary(keywords_lists)
    corpus_test = [dictionary.doc2bow(train_sentence) for train_sentence in keywords_lists]
    lda = models.ldamodel.LdaModel.load('')
    topics_test = lda.get_document_topics(corpus_test)
    for i in range(len(user_ids)):
        print('用户id为' + user_ids[i] + '的分类情况：')
        print(topics_test[i])
        print('参考的模型为：')
    for topic in lda.print_topics(num_topics=k1, num_words=k2):
        print(topic)
    return topics_test, user_ids
def output_varity():
    topics_test, user_ids = predict_model()
    yq_probability = []
    sa_probability = []
    hj_probability = []
    yj_probability = []
    user_yq = []
    user_sa = []
    user_yj = []
    user_hj = []
    ret = ['yq', 'sa', 'yj', 'kj']
    print('len(user_ids)', len(user_ids), 'len(topics_test)', len(topics_test))
    def camulate_gra(t, i):
        sum1 = 0
        sum2 = 0
        sum3 = 0
        sum4 = 0
        for j in range(0, t):
            try:
                k = int(topics_test[i][j][0])
            except:
                print("____k____", k)
            if k == n1 or k == n2:
                try:
                    print("sum1", topics_test[i][j][1])
                    sum1 = topics_test[i][j][1] + sum1
                except:
                    print("*****sum2****", sum2)
            elif k == n3 or k == n4 or k == n5:
                try:
                    sum2 = topics_test[i][j][1] + sum2
                except:
                    print("*****sum2****", sum2)
                # print("sum2", topics_test[i][j][1])
            elif k == n6 or k == n7:
                sum4 = topics_test[i][j][1] + sum4
            else:
                try:
                    sum3 = topics_test[i][j][1] + sum3
                except:
                    print("*****sum2****", sum2)
        return sum1, sum2, sum3, sum4
    length1 = int(len(user_ids))
    for i in range(0, length1):
        logging.info('topics_test[0]', topics_test[0], 'i', i)
        lenth = len(topics_test[i])
        count = i
        sum1, sum2, sum3, sum4 = camulate_gra(lenth, count)
        yq_probability.append(sum1)
        sa_probability.append(sum2)
        hj_probability.append(sum3)
        yj_probability.append(sum4)
    length2 = int(len(user_ids))
    for p in range(length2):
        print('用户id为' + user_ids[p] + '的topic：')
        print('yq：', yqi_probability[p], '\tsa：', shian_probability[p],
              '\tyj：', yijian_probability[p], '\tkj：', huanjing_probability[p])
        dic = {'yq': yiqing_probability[p], 'sa': shian_probability[p],
               'yj': yijian_probability[p], 'kj': huanjing_probability[p]}
        for i in range(len(ret)):
            if dic[ret[i]] > p1:
                if ret[i] == 'sa':
                    user_shian.append(user_ids[p])
            if dic[ret[i]] > p2:
                if ret[i] == 'yq':
                    user_yiqing.append(user_ids[p])
            if dic[ret[i]] > p3:
                if ret[i] == 'yj':
                    user_yijian.append(user_ids[p])
            if dic[ret[i]] > p4:
                if ret[i] == 'kj':
                    user_huanjing.append(user_ids[p])
    print('user_said', user_sa)
    print('user_yqid', user_yq)
    print('user_hj', user_hj)
    print('user_yj', user_yj)
    def conserve_user_id(path, list_id):
        conserve_path = path
        with open(conserve_path, 'w', encoding='utf-8') as f:
            print("list_id的长度：", len(list_id))
            for i in range(len(list_id)):
                f.write(str(list_id[i]))
                f.write('\n')
            f.close()
    conserve__path = ['user_sa.txt',
                      'user_yq.txt',
                      'user_hj.txt',
                      'user_yj.txt']
    user = [user_sa,
            user_yq,
            user_hj,
            user_yj]
    for i in range(len(user)):
        conserve_user_id(conserve__path[i], user[i])
if __name__ == '__main__':
    output_varity()