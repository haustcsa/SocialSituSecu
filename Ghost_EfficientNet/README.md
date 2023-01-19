#
代码使用简介

1.下载好数据集，创建dataset（train、val）文件夹按照数据集的文件结构进行摆放(即一个类别对应一个文件夹)，并且将训练以及预测脚本中的num_classes设置成你自己数据的类别数

2.在train.py脚本中将--data-path设置成dataset文件夹绝对路径

3.下载预训练权重，根据自己使用的模型下载对应预训练权重

4.在train.py脚本中将--weights参数设成下载好的预训练权重路径

5.设置好数据集的路径--data-path以及预训练权重的路径--weights就能使用train.py脚本开始训练了(训练过程中会自动生成class_indices.json文件)

6.在predict.py脚本中导入和训练脚本中同样的模型，并将model_weight_path设置成训练好的模型权重路径(默认保存在weights文件夹下)

7.在predict.py脚本中将img_path设置成你自己需要预测的图片绝对路径

8.设置好权重路径model_weight_path和预测的图片路径img_path就能使用predict.py脚本进行预测了


#### 源代码相关论文 
*[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
*[GhostNet: More Features from Cheap Operations](https://arxiv.org/abs/1911.11907)




