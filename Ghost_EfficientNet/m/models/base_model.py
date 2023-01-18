
import torch.nn as nn

from .MobileNet_v3 import MobileNetV3
from .ShuffleNet_v1 import shufflenet_g1
from .ShuffleNet_v2 import shufflenet_v2_x1_5
from .GhostNet import GhostNet
from .EfficientNet import efficientnet_b0,efficientnet_b1,efficientnet_b2,efficientnet_b3,efficientnet_b4,efficientnet_b5,efficientnet_b6,efficientnet_b7
from .EfficientNet_v2 import efficientnetv2_s,efficientnetv2_l,efficientnetv2_m
from .ConvNext import convnext_base,convnext_large,convnext_small,convnext_tiny,convnext_xlarge
from .MobileNetV3_Ghost import MobileNetV3_Ghost
from .MobileNetV3_Ghost221SE import MobileNetV3_Ghost221SE
from .MobileNetV3_Ghost112SE import MobileNetV3_Ghost112SE
from .ShuffleNetV2_linetoconv import ShuffleNetV2_linetoconv_x1_5


class BaseModel(nn.Module):
    def __init__(self, name, num_classes):
        super(BaseModel, self).__init__()
        # if name == 'AlexNet':
        #     self.base = alexnet(num_classes=num_classes)
        # elif name == 'VggNet':
        #     self.base = vgg16(num_classes=num_classes)
        # elif name == 'GoogleNet':
        #     self.base = GoogLeNet(num_classes=num_classes)
        # elif name == 'ResNet':
        #     self.base = resnet34(num_classes=num_classes)
        # elif name == 'ResNext':
        #     self.base = resnext50_32x4d(num_classes=num_classes)
        # elif name == 'DenseNet':
        #     self.base = densenet121(num_classes=num_classes)
        # elif name == 'MobileNet_v2':
        #     self.base = MobileNetV2(num_classes=num_classes)
        if name == 'MobileNet_v3':
            self.base = MobileNetV3(num_classes=num_classes)
            m = MobileNetV3(num_classes=num_classes)

        # elif name == 'ShuffleNet_v1':
        #     self.base = shufflenet_g1(num_classes=num_classes)
        elif name == 'shufflenetv2x15':
            self.base = shufflenet_v2_x1_5(num_classes=num_classes)
        elif name == 'EfficientNet_v1_B0':
            self.base = efficientnet_b0(num_classes=num_classes)
        # elif name == 'EfficientNet_v2_s':
        #     self.base = efficientnetv2_s(num_classes=num_classes)
        elif name == 'GhostNet':
            self.base = GhostNet(num_classes=num_classes)
        # elif name == 'ConvNext':
        #     self.base = convnext_tiny(num_classes=num_classes)
        elif name == 'MobileNetV3_Ghost':
            self.base = MobileNetV3_Ghost(num_classes=num_classes)
        elif name == 'MobileNetV3_Ghost221SE':
            self.base = MobileNetV3_Ghost221SE(num_classes=num_classes)
        elif name == 'MobileNetV3_Ghost112SE':
            self.base = MobileNetV3_Ghost112SE(num_classes=num_classes)
        elif name == 'ShuffleNetV2linetoconvx15':
            self.base = ShuffleNetV2_linetoconv_x1_5(num_classes=num_classes)
        else:
            raise ValueError('Input model name is not supported!!!')

    def forward(self, x):
        return self.base(x)
