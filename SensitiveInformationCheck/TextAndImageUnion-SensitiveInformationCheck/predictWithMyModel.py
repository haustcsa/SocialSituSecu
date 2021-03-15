import timeit
import numpy as np
import base64
from io import BytesIO
import logging
from keras.models import load_model
from keras.preprocessing import image
from keras import backend as K
from PIL import ImageFile
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

if K.image_data_format() == 'channels_first':
    input_shape = (3, 224, 224)
else:
    input_shape = (224, 224, 3)

## 定义log
logger = logging.getLogger(__name__)

# 返回一个编译好的模型
# 与之前那个相同
print("开始加载模型：")
# logger.info("开始加载模型")
starttime = timeit.default_timer()
model = load_model('data/modelFile/my_model.h5')
endtime = timeit.default_timer()
print("模型加载完成，用时为：", endtime - starttime)


# logger.info("模型加载完成，用时为：%s", endtime - starttime)

def predictWithImagePath(img_info):
    """
    根据输入图像path，来分析图像，并作出分类。
    :param filePath:图像路径
    :return:图像的类别
    """
    img_path = img_info["filePath"]
    type = img_info["type"]
    # 加载图像
    img = image.load_img(img_path, target_size=input_shape)
    # 图像预处理
    x = image.img_to_array(img) / 255.0  # 与训练一致
    x = np.expand_dims(x, axis=0)

    # 对图像进行分类
    preds = model.predict(x)  # Predicted: [[1.0000000e+00 1.4072199e-33 1.0080164e-22 3.4663230e-32]]
    print('Predicted:', preds)  # 输出预测概率
    predicted_class_indices = np.argmax(preds, axis=1)
    print('predicted_class_indices:', predicted_class_indices)  # 输出预测类别的int

    labels = {'political': 0, 'porn': 1, 'terrorism': 2}
    labels = dict((v, k) for k, v in labels.items())
    predicted_class = labels[predicted_class_indices[0]]
    # predictions = [labels[k] for k in predicted_class_indices]
    print("predicted_class :", predicted_class)
    if (preds[0][predicted_class_indices[0]]) <= 0.6:
        predicted_class = 'neutral'
        preds[0][predicted_class_indices[0]] = 0
    return predicted_class, preds[0][predicted_class_indices[0]]


def predictWithImageBase64(imageBase64String):
    """
    根据输入图像Base64，来分析图像，并作出分类。
    :param ImageBase64:图像Base64编码
    :return:图像的类别
    """
    try:
        # 加载图像 base64
        imageBase64String = imageBase64String.split(',')
        if (len(imageBase64String) > 1):
            imageBase64String = imageBase64String[1]
        else:
            imageBase64String = imageBase64String[0]

        imageBinaryData = base64.b64decode(imageBase64String)  # 解码base64
        imageData = BytesIO(imageBinaryData)  # 在内存中读取
        img = image.load_img(imageData, target_size=input_shape)  # 读取图片，并压缩至指定大小
        # 图像预处理
        x = image.img_to_array(img) / 255.0  # 与训练一致
        x = np.expand_dims(x, axis=0)
    except:
        return "98", "失败，解析 imageBase64String 参数的过程失败。", 0, 0

    # 对图像进行分类
    try:
        preds = model.predict(x)  # Predicted: [[1.0000000e+00 1.4072199e-33 1.0080164e-22 3.4663230e-32]]
    except:
        return "98", "失败，模型运行失败。", 0, 0
    print('Predicted:', preds)  # 输出预测概率
    logger.info('Predicted:%s', preds)
    predicted_class_indices = np.argmax(preds, axis=1)
    print('predicted_class_indices:', predicted_class_indices)  # 输出预测类别的int
    logger.info('predicted_class_indices:%s', predicted_class_indices)

    labels = {'neutral': 0, 'political': 1, 'porn': 2, 'terrorism': 3}
    labels = dict((v, k) for k, v in labels.items())
    predicted_class = labels[predicted_class_indices[0]]
    # predictions = [labels[k] for k in predicted_class_indices]
    print("predicted_class :", predicted_class)
    logger.info('predicted_class:%s', predicted_class)

    return "00", "成功", predicted_class, preds[0][predicted_class_indices[0]]


if __name__ == '__main__':
    img_path_list = []
    neutralCount = 0
    politicalCount = 0
    pornCount = 0
    terrorismCount = 0
    neutralNum = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    files = os.listdir("data/validation/")
    for file in files:
         if os.path.isdir("data/validation/" + file):
            for chlidFile in os.listdir("data/validation/" + file):
                if not chlidFile[0] == "." :
                    dict1 = dict()
                    dict1["type"] = file
                    dict1["filePath"] = "data/validation/" + file + "/" + chlidFile
                    img_path_list.append(dict1)
                    if file == "political":
                        politicalCount = politicalCount + 1
                    if file == "porn":
                        pornCount = pornCount + 1
                    if file == "terrorism":
                        terrorismCount = terrorismCount + 1
                    if file == "netural":
                        neutralCount = neutralCount + 1
    print(img_path_list)
    print("开始预测：")
    accuracyCount = 0
    accuracyTypeCount = {"political": 0, "porn": 1, "terrorism": 2}
    for i in img_path_list:
        starttime = timeit.default_timer()
        predictClass, predictAccuracy = predictWithImagePath(i)
        if i["type"] == "netural":

            if predictClass == "political" or predictClass == "porn" or predictClass == "terrorism":
                FP = FP + 1
            else:
                neutralNum = neutralNum + 1
                TN = TN + 1
            continue
        if predictClass == i["type"]:
            accuracyCount = accuracyCount + 1
            accuracyType = accuracyTypeCount[predictClass]
            accuracyTypeCount[predictClass] = accuracyType + 1
            TP = TP + 1
        else:
            FN = FN+ 1
        endtime = timeit.default_timer()
        print("单次调用模型预测时间为：", endtime - starttime)

    #print("************************** neutral *************************")
    #print("neutral的图片总数为:" + str(neutralCount) + "张,预测成功图片：" + str(neutralNum) + "张,准确率为:" + str(neutralNum / neutralCount))

    print("************************** political *************************")
    print("political的图片总数为:" + str(politicalCount) + "张,预测成功图片：" + str(accuracyTypeCount["political"]) + "张,准确率为:" + str(accuracyTypeCount["political"] / politicalCount))

    print("************************** porn *************************")
    print("porn的图片总数为:" + str(pornCount) + "张,预测成功图片：" + str(accuracyTypeCount["porn"]) + "张,准确率为:" + str(accuracyTypeCount["porn"] / pornCount))

    print("************************** terrorism *************************")
    print("terrorism的图片总数为:" + str(terrorismCount) + "张,预测成功图片：" + str(accuracyTypeCount["terrorism"]) + "张,准确率为:" + str(accuracyTypeCount["terrorism"] / terrorismCount))

    print("************************ 三种类型的统计*************************")
    print("三种类型的图片总数为：" + str(politicalCount + pornCount + terrorismCount) + "张，预测成功图片：" + str((accuracyTypeCount["political"] + accuracyTypeCount["porn"] +     accuracyTypeCount["terrorism"])) + "张，准确率为：" + str((accuracyTypeCount["political"] + accuracyTypeCount["porn"] + accuracyTypeCount["terrorism"]) / (politicalCount + pornCount + terrorismCount)))

    print("************************** 总共 *************************")
    A= TP/(TP+FP)
    P=(TP+TN) / (TP+FP+TN+FN)
    R = TP/(TP+FN)
    print("总共预测图片：" + str(TP+FP+TN+FN) + ",预测成功图片：" + str(TP+TN) + "张,总准确率为：" + str(P)+"总召回率为："+str(R)+",F值为："+str(2*A*R/(A+R)))