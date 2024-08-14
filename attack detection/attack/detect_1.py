from network import LeNet, CIFAR_LeNet
import os.path
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
import argparse
import random
from attack_utils import FGSM, DeepFool, CW, MJSMA, BIM
from tqdm import tqdm
from metric import discrete_entropy, two_quantization, four_quantization, six_quantization, cross_mask
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def detect_1(img):
    img = img.reshape(-1, img.shape[-1], img.shape[-1])
    entropy = discrete_entropy(img)
    if entropy <= 4:
        img_qua = two_quantization(img)
        return img_qua.reshape(1,-1,img.shape[-1], img.shape[-1])
    elif 5 >= entropy > 4:
        img_qua = four_quantization(img)
        return img_qua.reshape(1,-1,img.shape[-1], img.shape[-1])
    elif entropy > 5:
        img_qua = six_quantization(img)

        img_fil = cross_mask(img_qua)

        if torch.abs(img.sum() - img_qua.sum()) <= torch.abs(img.sum() - img_fil.sum()):
            return img_qua.reshape(1,-1,img.shape[-1], img.shape[-1])
        else:
            return img_fil.reshape(1,-1,img.shape[-1], img.shape[-1])


def test(net, attack, e, max_iter):
    positive = 0
    p = 1
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
            X,  SSIM, PSNR = BIM(net, X, e)
        elif attack == 'MJSMA':
            X ,  SSIM, PSNR= MJSMA(net, device, X, y, max_iter)

        X_ = detect_1(X.clone().detach())
        pred = net(X)
        pred_ = net(X_)

        if torch.argmax(pred) != y:
            p += 1
            if torch.argmax(pred_) != torch.argmax(pred):
                    positive += 1
        loop.set_description('Testing,%.2f'% (positive*100/p))

    acc = positive / p*100
    print('检测准确率:%.2f' % acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', default=1, type=int)
    parser.add_argument('-dataset', default='SVHN', type=str, help='MNIST, CIFAR10, SVHN')
    parser.add_argument('-net', default='lenet', type=str, help='lenet, CIFAR_lenet')
    parser.add_argument('-epochs', default=50, type=int)
    parser.add_argument('-lr', default=0.0005, type=float)
    parser.add_argument('-attack', default='DeepFool', type=str, help='FGSM, DeepFool, CW, BIM, MJSMA')
    parser.add_argument('-epsilon', default=0.3)
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

    test_dataloader = DataLoader(dataset=test_data, batch_size=args.batch_size)

    if args.net == 'lenet':
        net = LeNet(channels, size)
    elif args.net == 'CIFAR_lenet':
        net = CIFAR_LeNet(channels)
    else:
        print('您选择的网络在代码中不存在')

    net.to(device)

    if os.path.exists("model/%s.pth" % (args.net + '_' + args.dataset)):
        net.load_state_dict(torch.load("model/%s.pth" % (args.net + '_' + args.dataset)))
    else:
        print('已训练好的模型不存在')

    test(net, args.attack, args.epsilon, args.max_iter)











