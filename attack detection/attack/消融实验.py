from network import LeNet, FFR, FFR_conv, CIFAR_LeNet
import os.path
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
import argparse
import random
from tqdm import tqdm
from torchvision.transforms import Resize
from attack_utils import FGSM, DeepFool, BIM, MJSMA, CW
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def train(net, epochs):
    losses = []
    iteration = 0
    net.train()
    for epoch in range(epochs):
        loss_sum = 0
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, (X, y) in loop:
            X = X.to(device)
            pred = net(X)
            loss = loss_fn(pred, X)

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
    torch.save(train_net.state_dict(), "model/%s.pth" % (args.train_net + '_' + args.dataset))

    # 绘制损失函数曲线
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.plot(list(range(iteration)), losses)
    plt.savefig('result/loss.jpg')


def test(attack, e, max_iter):
    positive = 0
    p = 1
    train_net.eval()
    test_net.eval()
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
        X_ = X.clone().detach()
        if args.r_size:
            X_ = Resize((X_.shape[-1] - 8, X_.shape[-1] - 8))(X_)
        X_ = train_net(X_).detach()
        if args.r_size:
            X_ = Resize((X_.shape[-1] + 8, X_.shape[-1] + 8))(X_)
        pred = test_net(X)
        pred_ = test_net(X_)
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
    parser.add_argument('-train_net', default='ffr', type=str, help='ffr, ffr_conv')
    parser.add_argument('-test_net', default='lenet', type=str, help='lenet, CIFAR_lenet')
    parser.add_argument('-r_size', default=True, type=bool, help='是否使用resize')
    parser.add_argument('-epochs', default=5, type=int)
    parser.add_argument('-lr', default=0.001, type=float)
    parser.add_argument('-attack', default='FGSM', type=str, help='FGSM, DeepFool, CW, BIM, MJSMA')
    parser.add_argument('-epsilon', default=0.3)
    parser.add_argument('-max_iter', default=10)
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

    if args.train_net == 'ffr':
        train_net = FFR(channels)
        train_net.to(device)
    elif args.train_net == 'ffr_conv':
        train_net = FFR_conv(channels)
        train_net.to(device)
    if args.test_net == 'lenet':
        test_net = LeNet(channels, size)
        test_net.to(device)
    elif args.test_net == 'CIFAR_lenet':
        test_net = CIFAR_LeNet(channels)
        test_net.to(device)

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(params=train_net.parameters(), lr=args.lr)

    if os.path.exists("model/%s.pth" % (args.train_net + '_' + args.dataset)):
        train_net.load_state_dict(torch.load("model/%s.pth" % (args.train_net + '_' + args.dataset)))
    else:
        train(train_net, args.epochs)

    if os.path.exists("model/%s.pth" % (args.test_net + '_' + args.dataset)):
        test_net.load_state_dict(torch.load("model/%s.pth" % (args.test_net + '_' + args.dataset)))

    test(args.attack, args.epsilon, args.max_iter)











