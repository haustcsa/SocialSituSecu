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

def test(attack, e, max_iter):
    train_net.eval()
    test_net.eval()
    count = 0
    loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    for i, (X, y) in loop:
        X, y = X.to(device), y.to(device)
        if attack == 'FGSM':
            X, SSIM, PSNR = FGSM(test_net, X, y, e)
        elif attack == 'DeepFool':
            X, SSIM, PSNR = DeepFool(test_net, device, X, y, max_iter)
        elif attack == 'CW':
            X,  SSIM, PSNR = CW(test_net, device, X, y, max_iter)
        elif attack == 'BIM':
            X,  SSIM, PSNR = BIM(test_net, X, e)
        elif attack == 'MJSMA':
            X ,  SSIM, PSNR= MJSMA(test_net, device, X, y, max_iter)

        if args.detect == '1':
            out = detect_1(X)
        elif args.detect == '2':
            out = detect_2(X)
        elif args.detect == '3':
            out = detect_3(X)
        elif args.detect == 'ours':
            if args.r_size:
                X = Resize((X.shape[-1]-8, X.shape[-1]-8))(X)
            out = train_net(X).detach()
            if args.r_size:
                out = Resize((out.shape[-1]+8, out.shape[-1]+8))(X)
        else:
            out = X


        plt.imshow(out.reshape(28, 28, 1).detach().cpu())
        plt.axis('off')
        if count == 2:
            if args.detect:
                plt.savefig('result/MNIST_detect%s_%s.png' % (args.detect, args.attack), bbox_inches='tight', pad_inches=0)
            else:
                plt.savefig('result/MNIST_%s.png' % args.attack, bbox_inches='tight', pad_inches=0)
            plt.show()
            break
        count = count + 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='MNIST', type=str, help='MNIST, CIFAR10, SVHN')
    parser.add_argument('-train_net', default='ffr_single', type=str, help='ffr, ffr_single')
    parser.add_argument('-test_net', default='lenet', type=str, help='lenet, CIFAR_lenet')
    parser.add_argument('-r_size', default=False, type=bool, help='是否使用resize')
    parser.add_argument('-attack', default='none', type=str, help='FGSM, DeepFool, CW, BIM, MJSMA')
    parser.add_argument('-detect', default='none', type=str, help='1, 2, 3, ours')
    parser.add_argument('-epsilon', default=0.1)
    parser.add_argument('-max_iter', default=10)
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

    test(args.attack, args.epsilon, args.max_iter)











