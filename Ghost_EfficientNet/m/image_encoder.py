import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os, glob
import h5py
import time
from tqdm import tqdm


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        MobileNet = models.mobilenet_v3_small(pretrained=True)  # 使用与训练权重
        self.feature = MobileNet.features

    def forward(self, x):
        output = self.feature(x)  # 输出维度为（512*7*7）
        m = nn.MaxPool2d(7, stride=1)
        output = m(output)  # 对输出的维度进行平均池化->（512*1*1）
        output = output.view(output.size(0), -1)  # 512*1
        return output


model = Encoder()  # 定义模型


def extractor(img_path, net, use_gpu, file_name):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()]
    )  # 将图像转化为张量的形式，同时扩充224*224
    img = Image.open(img_path)  # 读取图像
    img = transform(img)
    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    if use_gpu:
        x = x.cuda()
        net = net.cuda()
    y = net(x).cpu()
    y = torch.squeeze(y)
    y_n = y.data.numpy()
    file_list = file_name
    return file_list, y_n


if __name__ == '__main__':
    data_dir = './ImgDB'  # 图像路径
    files_list = []
    names = []
    features = []
    x = os.walk(data_dir)
    for path, d, filelist in x:
        for filename in filelist:
            file_glob = os.path.join(path, filename)
            files_list.extend(glob.glob(file_glob))
    use_gpu = torch.cuda.is_available()
    pbar = tqdm(files_list)
    for x_path in files_list and pbar:
        file_name = x_path.split('/')[-1]
        file_names, ys = extractor(x_path, model, use_gpu, file_name)
        features.append(ys)
        names.append(file_names)
        pbar.set_description("Extracting features")
        # 设置进度条右边显示的信息
        pbar.set_postfix(Img_name=file_names)
        time.sleep(0.1)
    features = np.array(features)
    h5f = h5py.File('feature.h5', 'w')  # 创建h5数据库，用于存储数据
    h5f.create_dataset('tensor', data=features)  # 创建一个key，里面存储的是所有图像的特征
    h5f.create_dataset('name', data=names)  # 创建一个key，里面存储的是所有图像的名字
    h5f.close()
