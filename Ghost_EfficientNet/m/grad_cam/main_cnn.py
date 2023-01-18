import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from Mao_ConvNet.models.base_model import BaseModel
from torchvision import transforms
from Mao_ConvNet.grad_cam.utils import GradCAM, show_cam_on_image, center_crop_img
from Mao_ConvNet.utils import img_transform


def main(mobile_name, modelpath):
    device = torch.device("cpu")
    model = BaseModel(num_classes=4, name=mobile_name)
    model_weight_path = modelpath
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    print(model)

    target_layers = [model.base.large_bottleneck[-1]]

    data_transform = transforms.Compose([transforms.Resize(224),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    img_path = "微信截图_20220524161509.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = transforms.Compose([transforms.Resize([224,224])])(img)
    # img = center_crop_img(img, 224)
    plt.imshow(img)
    # [C, H, W]
    img_tensor = data_transform(img)


    img = np.array(img, dtype=np.uint8)


    print(img.shape)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = 2# tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]



    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    path = '../w/MobileNet_v3/MobileNet_v3_368_0.9341637010676157.pth'
    main("MobileNet_v3", path)
