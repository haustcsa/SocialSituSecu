from network import LeNet, CIFAR_LeNet
import os.path
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import Resize
from torch.utils.data import DataLoader
import argparse
import random
from attack_utils import FGSM, DeepFool, CW, MJSMA, BIM
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')


def bregman(image, mask, weight, eps=1e-3, max_iter=100):
    rows, cols, dims = image.shape
    rows2 = rows + 2
    cols2 = cols + 2
    total = rows * cols * dims
    shape_ext = (rows2, cols2, dims)

    u = torch.zeros(shape_ext)
    dx = torch.zeros(shape_ext)
    dy = torch.zeros(shape_ext)
    bx = torch.zeros(shape_ext)
    by = torch.zeros(shape_ext)

    u[1:-1, 1:-1] = image
    # reflect image
    u[0, 1:-1] = image[1, :]
    u[1:-1, 0] = image[:, 1]
    u[-1, 1:-1] = image[-2, :]
    u[1:-1, -1] = image[:, -2]

    i = 0
    rmse = torch.inf
    lam = 2 * weight
    norm = (weight + 4 * lam)

    while i < max_iter and rmse > eps:
        rmse = 0

        for k in range(dims):
            for r in range(1, rows + 1):
                for c in range(1, cols + 1):
                    uprev = u[r, c, k]

                    # forward derivatives
                    ux = u[r, c + 1, k] - uprev
                    uy = u[r + 1, c, k] - uprev

                    # Gauss-Seidel method
                    if mask[r - 1, c - 1]:
                        unew = (lam * (u[r + 1, c, k] +
                                       u[r - 1, c, k] +
                                       u[r, c + 1, k] +
                                       u[r, c - 1, k] +
                                       dx[r, c - 1, k] -
                                       dx[r, c, k] +
                                       dy[r - 1, c, k] -
                                       dy[r, c, k] -
                                       bx[r, c - 1, k] +
                                       bx[r, c, k] -
                                       by[r - 1, c, k] +
                                       by[r, c, k]
                                       ) + weight * image[r - 1, c - 1, k]
                                ) / norm
                    else:
                        # similar to the update step above, except we take
                        # lim_{weight->0} of the update step, effectively
                        # ignoring the l2 loss
                        unew = (u[r + 1, c, k] +
                                u[r - 1, c, k] +
                                u[r, c + 1, k] +
                                u[r, c - 1, k] +
                                dx[r, c - 1, k] -
                                dx[r, c, k] +
                                dy[r - 1, c, k] -
                                dy[r, c, k] -
                                bx[r, c - 1, k] +
                                bx[r, c, k] -
                                by[r - 1, c, k] +
                                by[r, c, k]
                                ) / 4.0
                    u[r, c, k] = unew

                    # update rms error
                    rmse += (unew - uprev) ** 2

                    bxx = bx[r, c, k]
                    byy = by[r, c, k]

                    # d_subproblem
                    s = ux + bxx
                    if s > 1 / lam:
                        dxx = s - 1 / lam
                    elif s < -1 / lam:
                        dxx = s + 1 / lam
                    else:
                        dxx = 0
                    s = uy + byy
                    if s > 1 / lam:
                        dyy = s - 1 / lam
                    elif s < -1 / lam:
                        dyy = s + 1 / lam
                    else:
                        dyy = 0

                    dx[r, c, k] = dxx
                    dy[r, c, k] = dyy

                    bx[r, c, k] += ux - dxx
                    by[r, c, k] += uy - dyy

        rmse = torch.sqrt(rmse / total)
        i += 1

    return torch.squeeze(torch.asarray(u[1:-1, 1:-1]))


def defend_tv(input, keep_prob=0.5, lambda_tv=0.03):
    mask = np.random.uniform(size=input.shape[:2])
    mask = mask < keep_prob
    return bregman(input, mask, weight=2.0 / lambda_tv)


def detect_3(img):
    i_shape = img.shape
    if img.shape[1] == 1:
        img = img.reshape(img.shape[3], img.shape[3], 1)
        img = defend_tv(img.cpu())
        img = img.reshape(i_shape)
        img = img.cuda()
    else:
        img = torch.cat((img[:, 0].reshape(img.shape[3], img.shape[3], 1), img[:, 1].reshape(img.shape[3], img.shape[3], 1),
                         img[:, 2].reshape(img.shape[3], img.shape[3], 1)), dim=2)
        img = defend_tv(img.cpu())
        img = torch.cat(
            (img[:, :, 0].reshape(1, img.shape[0], img.shape[0]), img[:, :, 1].reshape(1, img.shape[0], img.shape[0]),
             img[:, :, 2].reshape(1, img.shape[0], img.shape[0])), dim=0)
        img = img.reshape(i_shape)
        img = img.cuda()

    return img


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

        X_ = detect_3(X.clone().detach())
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
    parser.add_argument('-dataset', default='CIFAR10', type=str, help='MNIST, CIFAR10, SVHN')
    parser.add_argument('-net', default='CIFAR_lenet', type=str, help='lenet, CIFAR_lenet')
    parser.add_argument('-epochs', default=50, type=int)
    parser.add_argument('-lr', default=0.0005, type=float)
    parser.add_argument('-attack', default='MJSMA', type=str, help='FGSM, DeepFool, CW, BIM, MJSMA')
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











