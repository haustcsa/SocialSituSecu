import numpy as np

from network import LeNet, FFR, FFR_single, CIFAR_LeNet
import os.path
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
import argparse
import random
from tqdm import tqdm
from detect_1 import detect_1
from detect_2 import detect_2
from detect_3 import detect_3
from torchvision.transforms import Resize
from attack_utils import FGSM, DeepFool, BIM, MJSMA, CW
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def get_fig(x, y, attack):
    plt.figure(dpi=1000)
    plt.plot(x, y[:10], alpha=0.7, c='r', label='ANR')
    plt.plot(x, y[10:20], alpha=0.7, c='g', label='RAN')
    plt.plot(x, y[20:30], alpha=0.7, c='b', label='TVM')
    plt.plot(x, y[30:40], alpha=0.7, c='c', label='FDR')
    plt.plot(x, y[40:], alpha=0.7, c='m', label='FDR_GM')

    if attack == 'FGSM' or attack == 'BIM':
        plt.xticks([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3])
        plt.xlabel('Epsilon')
    elif attack == 'DeepFool' or attack == 'MJSMA':
        plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        plt.xlabel('Iterations')
    else:
        plt.xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('result/top1/%s_%s.png' % (args.dataset, attack))


def test_a(attack, detect, e=0, max_iter=0):
    train_net.eval()
    test_net.eval()
    a = 0
    al = 0
    count = -1
    loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader))

    for i, (X, y) in loop:

        count += 1
        X, y = X.to(device), y.to(device)

        if attack == 'FGSM':
            X, SSIM, PSNR = FGSM(test_net, X, y, e)
        elif attack == 'DeepFool':
            X, SSIM, PSNR = DeepFool(test_net, device, X, y, max_iter)
        elif attack == 'CW':
            X, SSIM, PSNR = CW(test_net, device, X, y, max_iter)
        elif attack == 'BIM':
            X, SSIM, PSNR = BIM(test_net, X, e)
        elif attack == 'MJSMA':
            X, SSIM, PSNR = MJSMA(test_net, device, X, y, max_iter)

        if detect == '1':
            out = detect_1(X)
        elif detect == '2':
            out = detect_2(X)
        elif detect == '3':
            out = detect_3(X)
        elif detect == 'ours1':
            out = train_net(X).detach()
        elif detect == 'ours2':
            if X.shape[-1] == 28:
                X = Resize((20, 20))(X)
                out = train_net(X).detach()
                out = Resize((28, 28))(out)
            if X.shape[-1] == 32 and args.test_net == 'lenet':
                X = Resize((18, 18))(X)
                out = train_net(X).detach()
                out = Resize((32, 32))(out)
            if X.shape[-1] == 32 and args.test_net == 'CIFAR_lenet':
                X = Resize((36, 36))(X)
                out = train_net(X).detach()
                out = Resize((32, 32))(out)


        pred = test_net(X)
        pred_ = test_net(out)

        if torch.argmax(pred) != y:
            al += 1
            if torch.argmax(pred_) != torch.argmax(pred):
                    a += 1
        if count == 100:
            return a, al


def test_b(detect):
    train_net.eval()
    test_net.eval()
    a = 0
    al = 0
    count = -1
    loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader))

    for i, (X, y) in loop:

        count += 1
        X, y = X.to(device), y.to(device)

        if detect == '1':
            out1 = detect_1(X)
        elif detect == '2':
            out1 = detect_2(X)
        elif detect == '3':
            out1 = detect_3(X)
        elif detect == 'ours1':
            out1 = train_net(X).detach()
        elif detect == 'ours2':
            if X.shape[-1] == 28:
                X = Resize((20, 20))(X)
                out1 = train_net(X).detach()
                out1 = Resize((28, 28))(out1)
            if X.shape[-1] == 32 and args.test_net == 'lenet':
                X = Resize((18, 18))(X)
                out1 = train_net(X).detach()
                out1 = Resize((32, 32))(out1)
            if X.shape[-1] == 32 and args.test_net == 'CIFAR_lenet':
                X = Resize((36, 36))(X)
                out1 = train_net(X).detach()
                out1 = Resize((32, 32))(out1)

        pred = test_net(X)
        pred_ = test_net(out1)

        if torch.argmax(pred) == y:
            al += 1
            if torch.argmax(pred_) == torch.argmax(pred):
                a += 1
        if count == 100:
            return a, al



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='CIFAR10', type=str, help='MNIST, CIFAR10, SVHN')
    parser.add_argument('-train_net', default='ffr', type=str, help='ffr, ffr_single')
    parser.add_argument('-test_net', default='CIFAR_lenet', type=str, help='lenet, CIFAR_lenet')
    parser.add_argument('-attack', default=['FGSM', 'DeepFool', 'CW', 'BIM', 'MJSMA'])
    parser.add_argument('-detect', default=['1', '2', '3', 'ours1', 'ours2'])
    parser.add_argument('-epsilon', default=[0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3])
    parser.add_argument('-max_iter', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    args = parser.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    torch.cuda.manual_seed_all(0)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('运行设备：%s' % device)

    if args.dataset == 'MNIST':
        test_data = torchvision.datasets.MNIST(root='dataset', train=False, download=False, transform=transforms.ToTensor())
        channels = 1
        size = 256
    elif args.dataset == 'CIFAR10':
        test_data = torchvision.datasets.CIFAR10(root='dataset', train=False, download=False, transform=transforms.ToTensor())
        channels = 3
        size = 256
    elif args.dataset == 'SVHN':
        test_data = torchvision.datasets.SVHN(root='dataset', split='test', download=False, transform=transforms.ToTensor())
        channels = 3
        size = 256
    else:
        print('您选择的数据集在代码中不存在')

    test_dataloader = DataLoader(dataset=test_data, batch_size=1)

    if args.train_net == 'ffr':
        train_net = FFR(channels)
        train_net.to(device)
    elif args.train_net == 'ffr_single':
        train_net = FFR_single(channels)
        train_net.to(device)

    if args.test_net == 'lenet':
        test_net = LeNet(channels, size)
        test_net.to(device)
    elif args.test_net == 'CIFAR_lenet':
        test_net = CIFAR_LeNet(channels)
        test_net.to(device)

    if os.path.exists("model/%s.pth" % (args.train_net + '_' + args.dataset)):
        train_net.load_state_dict(torch.load("model/%s.pth" % (args.train_net + '_' + args.dataset)))

    if os.path.exists("model/%s.pth" % (args.test_net + '_' + args.dataset)):
        test_net.load_state_dict(torch.load("model/%s.pth" % (args.test_net + '_' + args.dataset)))

    for attack in args.attack:
        y = []
        for detect in args.detect:
            if attack == 'FGSM' or attack == 'BIM':
                for e in args.epsilon:
                    p1, al1 = test_a(attack, detect, e=e)
                    p2, al2 = test_b(detect)
                    acc = (p1 + p2)/(al1 + al2)
                    y.append(acc)
            elif attack == 'DeepFool' or attack == 'MJSMA':
                for max_iter in args.max_iter:
                    p1, al1 = test_a(attack, detect, max_iter=max_iter)
                    p2, al2 = test_b(detect)
                    acc = (p1 + p2)/(al1 + al2)
                    y.append(acc)
            else:
                for max_iter in args.max_iter:
                    p1, al1 = test_a(attack, detect, max_iter=max_iter*10)
                    p2, al2 = test_b(detect)
                    acc = (p1 + p2)/(al1 + al2)
                    y.append(acc)

        if attack == 'FGSM' or attack == 'BIM':
            x = [0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3]
        elif attack == 'DeepFool' or attack == 'MJSMA':
            x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        else:
            x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        #get_fig(x, y, attack)
        np.savetxt('result/top1/%s_%s.csv' % (args.dataset, attack), y, fmt='%.3f')












