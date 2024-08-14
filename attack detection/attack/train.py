from network import LeNet, CIFAR_LeNet
import os.path
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
import argparse
import random
from tqdm import tqdm

from attack_utils import FGSM, DeepFool, BIM, MJSMA, CW
import sys
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')


def train(net, epochs):
    losses = []
    iteration = 0
    net.train()
    for epoch in range(epochs):
        loss_sum = 0
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, (X, y) in loop:
            X, y = X.to(device), y.to(device)

            pred = net(X)
            loss = loss_fn(pred, y)

            loss_sum += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_description(f'Epoch [{epoch}/{epochs}]')
            loop.set_postfix(loss=loss.item())
        mean_loss = loss_sum / len(train_dataloader.dataset)
        losses.append(mean_loss)
        iteration += 1

    # 训练完毕保存最后一轮训练的模型
    torch.save(net.state_dict(), "model/%s.pth" % (args.net + '_' + args.dataset))

    # 绘制损失函数曲线
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.plot(list(range(iteration)), losses)
    plt.savefig('result/loss.jpg')


class Logger(object):

    def __init__(self, filename='default.log'):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

import time
t = time.strftime("-%Y%m%d-%H%m%S", time.localtime())  # 时间
filename = '1og' + t + '.txt'  # 文件名

log = Logger(filename)  # 保存的文件名
sys.stdout = log

def test(net, attack, e, max_iter):
    SSIM_all = 0.
    PSNR_all = 0.
    positive = 0
    negative = 0
    net.eval()
    loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    for i, (X, y) in loop:
        X, y = X.to(device), y.to(device)
        if attack == 'FGSM':
            X, SSIM, PSNR = FGSM(net, X, y, e)
        elif attack == 'DeepFool':
            X, SSIM, PSNR = DeepFool(net, device, X, y, max_iter)
        elif attack == 'CW':
            X,  SSIM, PSNR = CW(net, device, X, y, max_iter)
        elif attack == 'BIM':
            X, SSIM, PSNR = BIM(net, X, e)
        elif attack == 'MJSMA':
            X, SSIM, PSNR = MJSMA(net, device, X, y, max_iter)

        SSIM_all = SSIM_all + SSIM
        PSNR_all = PSNR_all + PSNR

        pred = net(X)

        for item in zip(pred, y):
            if torch.argmax(item[0]) == item[1]:
                positive += 1
            else:
                negative += 1
        loop.set_description('准确率:%.2f, SSIM:%.2f, PSNR:%.2f' % (positive / (positive + negative) * 100, SSIM_all / 100, PSNR_all / 100))
    acc = positive / (positive + negative)*100

    print('准确率:%.2f, SSIM:%.2f, PSNR:%.2f' % (acc, SSIM_all / len(test_dataloader), PSNR_all / len(test_dataloader)) )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', default=2048, type=int)
    parser.add_argument('-dataset', default='MNIST', type=str, help='MNIST, CIFAR10, SVHN')
    parser.add_argument('-net', default='lenet', type=str, help='lenet,CIFAR_lenet')
    parser.add_argument('-epochs', default=20, type=int)
    parser.add_argument('-lr', default=0.0005, type=float)
    parser.add_argument('-attack', default='FGSM', type=str, help='FGSM, DeepFool, CW, BIM, MJSMA')
    parser.add_argument('-epsilon', default=0.3)
    parser.add_argument('-max_iter', default=1000)
    args = parser.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    torch.cuda.manual_seed_all(0)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('运行设备：%s' % device)

    if args.dataset == 'MNIST':
        train_data = torchvision.datasets.MNIST(root='dataset', train=True, download=False, transform=transforms.ToTensor())
        test_data = torchvision.datasets.MNIST(root='dataset', train=False, download=False, transform=transforms.ToTensor())
        channels = 1
        size = 256
    elif args.dataset == 'CIFAR10':
        train_data = torchvision.datasets.CIFAR10(root='dataset', train=True, download=False, transform=transforms.ToTensor())
        test_data = torchvision.datasets.CIFAR10(root='dataset', train=False, download=False, transform=transforms.ToTensor())
        channels = 3
        size = 256
    elif args.dataset == 'SVHN':
        train_data = torchvision.datasets.SVHN(root='dataset', split='train', download=False, transform=transforms.ToTensor())
        test_data = torchvision.datasets.SVHN(root='dataset', split='test', download=False, transform=transforms.ToTensor())
        channels = 3
        size = 256
    else:
        print('您选择的数据集在代码中不存在')
    train_dataloader = DataLoader(dataset=train_data, batch_size=args.batch_size)
    test_dataloader = DataLoader(dataset=test_data, batch_size=args.batch_size)
    if args.net == 'lenet':
        net = LeNet(channels, size)
    elif args.net == 'CIFAR_lenet':
        net = CIFAR_LeNet(channels)
    else:
        print('您选择的网络在代码中不存在')

    net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=args.lr)

    if os.path.exists("model/%s.pth" % (args.net + '_' + args.dataset)):
        net.load_state_dict(torch.load("model/%s.pth" % (args.net + '_' + args.dataset)))
    else:
        train(net, args.epochs)

    test(net, args.attack, args.epsilon, args.max_iter)











