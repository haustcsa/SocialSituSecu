import torch
from torch import nn
from torchvision.transforms import Resize
import torch.fft as fft

# 正常的分类模型
class LeNet(nn.Module):
    def __init__(self, channels, size):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=7, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2)

        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(size, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# 正常的分类模型——用于CIFAR10数据集的
class CIFAR_LeNet(nn.Module):
    def __init__(self, channels):
        super(CIFAR_LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2)

        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1000, 10)
        )

    def forward(self, x):

        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# FFR模型-用于MNIST
class FFR_single(nn.Module):
    def __init__(self, channels):
        super(FFR_single, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, channels, 1),
            nn.Tanh()
        )

        self.fft = nn.Sequential(
            nn.Conv2d(32, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1),
        )

    def forward(self, x):
        _, _, H, W = x.shape
        feature = self.encoder(x)

        # 傅里叶变换
        f_fft = fft.rfft2(feature)   # 8, 14, 14

        # 虚部
        f_imag = f_fft.imag  # 8, 14, 8
        # 实部
        f_real = f_fft.real  # 8, 14, 8

        feature_fft = torch.cat([f_real, f_imag], dim=1)  # 16, 14, 8
        feature_fft = self.fft(feature_fft)  # 8, 14, 8


        # 傅里叶逆变换
        f_real, f_imag = torch.chunk(feature_fft, 2, dim=1)  # 8, 14, 8
        feature = torch.complex(f_real, f_imag)  # 8, 14, 8

        feature = torch.fft.irfft2(feature)  # 8, 14, 14
        out = self.decoder(feature)
        return out




# FFR模型-用于CIFAR10, SVHN
class FFR(nn.Module):
    def __init__(self, channels):
        super(FFR, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.Tanh()
        )

        self.fft = nn.Sequential(
            nn.Conv2d(128, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 1),
        )

    def forward(self, x):
        _, _, H, W = x.shape
        feature = self.encoder(x)

        # 傅里叶变换
        f_fft = fft.rfft2(feature)   # 8, 14, 14

        f_imag = f_fft.imag  # 8, 14, 8
        f_real = f_fft.real  # 8, 14, 8

        feature_fft = torch.cat([f_real, f_imag], dim=1)  # 16, 14, 8
        feature_fft = self.fft(feature_fft)  # 8, 14, 8


        # 傅里叶逆变换
        f_real, f_imag = torch.chunk(feature_fft, 2, dim=1)  # 8, 14, 8
        feature = torch.complex(f_real, f_imag)  # 8, 14, 8

        feature = torch.fft.irfft2(feature)  # 8, 14, 14
        out = self.decoder(feature)
        return out

# FFR模型-用于消融实验
class FFR_conv(nn.Module):
    def __init__(self, channels):
        super(FFR_conv, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.Tanh()
        )

        self.fft = nn.Sequential(
            nn.ReLU(),
        )

    def forward(self, x):
        _, _, H, W = x.shape
        feature = self.encoder(x)

        # 傅里叶变换
        f_fft = fft.rfft2(feature)   # 8, 14, 14

        f_imag = f_fft.imag  # 8, 14, 8
        f_real = f_fft.real  # 8, 14, 8

        feature_fft = torch.cat([f_real, f_imag], dim=1)  # 16, 14, 8
        feature_fft = self.fft(feature_fft)  # 8, 14, 8


        # 傅里叶逆变换
        f_real, f_imag = torch.chunk(feature_fft, 2, dim=1)  # 8, 14, 8
        feature = torch.complex(f_real, f_imag)  # 8, 14, 8

        feature = torch.fft.irfft2(feature)  # 8, 14, 14
        out = self.decoder(feature)
        return out


if __name__ == '__main__':
    x = torch.ones([1, 1, 28, 28])
    net = FFR_single(1)
    print(net(x).shape)
