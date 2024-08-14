import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


def psnr(img1, img2):
    mse = torch.mean((img1-img2)**2)
    if mse == 0:
        return float('inf')
    else:
        return 20*torch.log10(255/torch.sqrt(mse))


def gray_entropy(img):
    hist = torch.histc(img, bins=256)
    probability = hist / float(img.shape[-1] * img.shape[-1])
    entropy = -torch.sum(probability * torch.log2(probability + 1e-7))
    #print("灰度图片的像素熵为：", entropy)
    return entropy

def discrete_entropy(img):
    if img.shape[0] == 3:

        red_channel = img[0, :, :]
        green_channel = img[1, :, :]
        blue_channel = img[2, :, :]
        red_entropy = gray_entropy(red_channel)
        green_entropy = gray_entropy(green_channel)
        blue_entropy = gray_entropy(blue_channel)
        total_entropy = (red_entropy + green_entropy + blue_entropy) / 3
    elif img.shape[0] == 1:
        total_entropy = gray_entropy(img)
    else:
        print('图片通道数量错误')

    # print("彩色图片的像素熵为：", total_entropy)

    return total_entropy


def two_quantization(img):
    for k in range(img.shape[0]):
        for i in range(img.shape[1]):
            for j in range(img.shape[2]):
                img[k][i][j] = int(img[k][i][j]/0.5) * 0.5

    return img


def four_quantization(img):
    for k in range(img.shape[0]):
        for i in range(img.shape[1]):
            for j in range(img.shape[2]):
                img[k][i][j] = int(img[k][i][j]/0.25) * 0.25
    return img


def six_quantization(img):
    for k in range(img.shape[0]):
        for i in range(img.shape[1]):
            for j in range(img.shape[2]):
                img[k][i][j] = int(img[k][i][j]/0.166666667) * 0.166666667
    return img


def cross_mask(img):
    channel = img.shape[0]
    sharp_kernel = torch.cuda.FloatTensor(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
             0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]).reshape(1, 1, 7, 7)
    if channel == 3:
        sharp_kernel = torch.cat([sharp_kernel, sharp_kernel.clone(), sharp_kernel.clone()], dim=1)
    conv2d = torch.nn.Conv2d(channel, channel, (7, 7), bias=False, padding=3, padding_mode='reflect').cuda()  # 设置卷积网络
    conv2d.weight.data = sharp_kernel  # 初始化weight
    conv2d.weight.requires_grad = False

    return conv2d(img)




