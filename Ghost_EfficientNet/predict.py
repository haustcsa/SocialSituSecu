import os
import json
import time
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from Mao_ConvNet.models.base_model import BaseModel
from Mao_ConvNet.utils import img_transform

Models = {0: "AlexNet",
          1: "VggNet",
          2: "GoogleNet",
          3: "ResNet",
          4: "ResNext",
          5: "DenseNet",
          6: "MobileNet_v2",
          7: "MobileNet_v3",
          8: "ShuffleNet_v1",
          9: "ShuffleNet_v2",
          10: "EfficientNet_v1_B0",
          11: "EfficientNet_v2_s",
          12: "GhostNet",
          13: "ConvNext"
          }


def cmp(x):
    return int(str(x).split("_")[-2])


def main(modelname, modelpath):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    data_transform = img_transform(modelname)["val"]

    # load image
    imgs_root = "./img"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root)]
    # img_path = "./img/img (4).jpg"
    # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # img = Image.open(img_path)
    # img = img.convert("RGB")
    # plt.imshow(img)
    # [N, C, H, W]
    # img = data_transform(img)
    # # expand batch dimension
    # img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = BaseModel(num_classes=4, name=modelname).to(device)
    # load model weights
    model_weight_path = modelpath
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    batch_size = len(img_path_list)  # 每次预测时将多少张图片打包成一个batch
    with torch.no_grad():
        start_time = time.time()
        for ids in range(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                img = Image.open(img_path).convert("RGB")
                img = data_transform(img)
                img_list.append(img)

            # batch img
            # 将img_list列表中的所有图像打包成一个batch
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)

            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                print("image: {}  class: {}  prob: {:.3}".format(img_path_list[ids * batch_size + idx],
                                                                 class_indict[str(cla.numpy())],
                                                                 pro.numpy()))
        end_time = time.time()
        use_time = end_time - start_time
        num = (len(img_path_list) // batch_size) * batch_size
        print(
            f"预测{num}张图片, batch:{batch_size}, 耗时{use_time}s, 平均耗时{use_time / num}s/张, {use_time / (len(img_path_list) // batch_size)}s/batch")


if __name__ == '__main__':
    path = "./w"

    dirs = [dirs_ for dirs_ in os.listdir(path) if os.path.isdir(os.path.join(path, dirs_))]
    for d in dirs:
        model_path = os.path.join("./w", d)
        model_list = [i for i in os.listdir(model_path) if i.endswith(".pth")]
        if len(model_list) == 0:
            continue
        model_list.sort(key=cmp)
        print()
        main(d, os.path.join(model_path, model_list[-1]))
