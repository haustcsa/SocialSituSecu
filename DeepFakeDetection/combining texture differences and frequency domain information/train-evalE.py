import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from einops import rearrange, repeat
from torch import nn, einsum
import torch.nn as nn
import torch.nn.functional as F
from random import random, randint, choice
from vit_pytorch import ViT
import numpy as np
import os
import json
from multiprocessing.pool import Pool
from functools import partial
from multiprocessing import Manager
from progress.bar import ChargingBar
from cross_efficient_vit import CrossEfficientViT
import uuid
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import accuracy_score, roc_auc_score
import cv2
from transforms.albu import IsotropicResize
import glob
import pandas as pd
from tqdm import tqdm
from utils import get_method, check_correct, resize, shuffle_dataset, get_n_params
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import lr_scheduler
import collections
from deepfakes_dataset import DeepFakesDataset
import math
import yaml
import argparse
from datetime import datetime
from sklearn import metrics
from sklearn.metrics import auc

# BASE_DIR = 'F:\FFtest'
# DATA_DIR = os.path.join(BASE_DIR, "dataset")
# TRAINING_DIR = os.path.join(DATA_DIR, "training_set")
# VALIDATION_DIR = os.path.join(DATA_DIR, "validation_set")
# TEST_DIR = os.path.join(DATA_DIR, "test_set")
# MODELS_PATH = "F:\Com_T_pth"
# METADATA_PATH = os.path.join(BASE_DIR, "data/metadata")  # Folder containing all training metadata for DFDC dataset
# VALIDATION_LABELS_PATH = os.path.join(DATA_DIR, "dfdc_val_labels22.csv")


BASE_DIR = 'E:\DFDC'
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TRAINING_DIR = os.path.join(DATA_DIR, "training_set")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation_set")
TEST_DIR = os.path.join(DATA_DIR, "test_set")
MODELS_PATH = "F:\Com_T_pth"
METADATA_PATH = os.path.join(BASE_DIR, "data/metadata")  # Folder containing all training metadata for DFDC dataset
VALIDATION_LABELS_PATH = os.path.join(DATA_DIR, "dfdc_val_labels22.csv")


def read_frames(video_path, train_dataset, validation_dataset):
    # video_path就是遍历train_dataset、validation_dataset中的子文件夹
    # Get the video label based on dataset selected
    # method = get_method(video_path, DATA_DIR)
    if TRAINING_DIR in video_path:
        label = 1
        if "Original" in video_path:
            label = 0
        elif "DFDC" in video_path:
            # for json_path in glob.glob(os.path.join(METADATA_PATH, "*.json")):
            for json_path in glob.glob(r"E:\DFDC\dataset\metadata\metadata.json"):
                with open(json_path, "r") as f:
                    metadata = json.load(f)
                video_folder_name = os.path.basename(video_path)

                video_key = video_folder_name + ".mp4"
                if video_key in metadata.keys():
                    item = metadata[video_key]
                    label = item.get("label", None)
                    if label == "FAKE":
                        label = 1.
                    else:
                        label = 0.
                    break
                else:
                    label = None
        else:
            label = 1.
        if label == None:
            print("NOT FOUND", video_path)
    else:
        # if "Original" in video_path:
        #     label = 0.
        # elif "DFDC" in video_path:
        #     val_df = pd.DataFrame(pd.read_csv(VALIDATION_LABELS_PATH))
        #     video_folder_name = os.path.basename(video_path)
        #     video_key = video_folder_name + ".mp4"
        #     label = val_df.loc[val_df['filename'] == video_key]['label'].values[0]
        # else:
        #     label = 1.
        if "Original" in video_path:
            label = 0
        elif "DFDC" in video_path:
            # for json_path in glob.glob(os.path.join(METADATA_PATH, "*.json")):
            for json_path in glob.glob(r"E:\DFDC\dataset\metadata\metadata.json"):
                with open(json_path, "r") as f:
                    metadata = json.load(f)
                video_folder_name = os.path.basename(video_path)

                video_key = video_folder_name + ".mp4"
                if video_key in metadata.keys():
                    item = metadata[video_key]
                    label = item.get("label", None)
                    if label == "FAKE":
                        label = 1.
                    else:
                        label = 0.
                    break
                else:
                    label = None
        else:
            label = 1.
        if label == None:
            print("NOT FOUND", video_path)

    # Calculate the interval to extract the frames 计算提取帧的间隔
    with open("F:\DeepFake\Vit_CNN\cross-efficient-vit\configs/architecture.yaml", 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    # print('video_path:--', video_path)
    # print('label:--', label)
    frames_number = len(os.listdir(video_path))

    if label == 0:
        min_video_frames = max(int(config['training']['frames-per-video'] * config['training']['rebalancing-real']),
                               1)  # Compensate unbalancing
        # print('min_video_frames label=0:',min_video_frames)
    else:
        min_video_frames = max(int(config['training']['frames-per-video'] * config['training']['rebalancing-fake']), 1)
        # print('min_video_frames label=1--:', min_video_frames)

        # if VALIDATION_DIR in video_path:
        #     min_video_frames = int(max(min_video_frames / 8, 2))
    frames_interval = int(frames_number / min_video_frames)
    frames_paths = os.listdir(video_path)
    frames_paths_dict = {}

    # Group the faces with the same index, reduce probabiity to skip some faces in the same video
    # 将具有相同索引的人脸分组，降低跳过同一视频中某些人脸的可能性
    for path in frames_paths:
        for i in range(0, 1):
            if "_" + str(i) in path:
                if i not in frames_paths_dict.keys():
                    frames_paths_dict[i] = [path]
                else:
                    frames_paths_dict[i].append(path)
    # Select only the frames at a certain interval
    # 仅选择特定间隔的帧
    if frames_interval > 0:
        for key in frames_paths_dict.keys():
            if len(frames_paths_dict) > frames_interval:
                frames_paths_dict[key] = frames_paths_dict[key][::frames_interval]

            frames_paths_dict[key] = frames_paths_dict[key][:min_video_frames]
    # Select N frames from the collected ones
    # 从收集的帧中选择N帧
    # print('frames_paths_dict.keys():---',frames_paths_dict.keys())
    for key in frames_paths_dict.keys():
        for index, frame_image in enumerate(frames_paths_dict[key]):
            # image = transform(np.asarray(cv2.imread(os.path.join(video_path, frame_image))))
            image = cv2.imread(os.path.join(video_path, frame_image))
            if image is not None:
                if TRAINING_DIR in video_path:
                    train_dataset.append((image, label))
                else:
                    validation_dataset.append((image, label))


# Main body
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=5, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--workers', default=1, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--resume', default="F:\Com_T_pth\efficientnet_checkpoint1_All",
                        type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none).')
    parser.add_argument('--dataset', type=str, default='Deepfakes',  # All
                        help="Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures|All)")
    parser.add_argument('--max_videos', type=int, default=-1,
                        help="Maximum number of videos to use for training (default: all).")
    parser.add_argument('--config', type=str,
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--efficient_net', type=int, default=0,
                        help="Which EfficientNet version to use (0 or 7, default: 0)")
    parser.add_argument('--patience', type=int, default=5,
                        help="How many epochs wait before stopping for validation loss not improving.")

    opt = parser.parse_args()
    print("opt的值")
    print(opt)

    now = datetime.now()
    print("开始时间：", now.strftime('%Y-%m-%d %H:%M:%S'))

    # with open(opt.config, 'r') as ymlfile:
    with open("F:\DeepFake\Vit_CNN\cross-efficient-vit\configs/architecture.yaml", 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    model = CrossEfficientViT(config=config)
    # model.train()
    model.eval()

    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['lr'],
                                weight_decay=config['training']['weight-decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'],
                                    gamma=config['training']['gamma'])
    starting_epoch = 0
    if os.path.exists(opt.resume):
        model.load_state_dict(torch.load(opt.resume))
        starting_epoch = int(opt.resume.split("checkpoint")[1].split("_")[0]) + 1
    else:
        print("No checkpoint loaded.")

    print("Model Parameters:", get_n_params(model))

    # READ DATASET
    if opt.dataset != "All":
        #folders = ["Original", opt.dataset]
        folders = ["DFDC"]
    else:
        folders = ["Original", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]

    sets = [TRAINING_DIR, VALIDATION_DIR]

    '''
    paths = []被创建后，遍历set，也就是上面的sets = [TRAINING_DIR, VALIDATION_DIR]
    之后paths中存储的是training_set以及validation_set中的所有子文件夹名，例如
    'G:\\DFDC\\dfdc_train_Part_49\\dataset\\training_set\\DFDC\\aahkgsltxw'
    'G:\\DFDC\\dfdc_train_Part_49\\dataset\\validation_set\\DFDC\\prqennmmxg'
    '''
    paths = []
    for dataset in sets:
        for folder in folders:
            subfolder = os.path.join(dataset, folder)  # 数据集的设置，直接设置为DFDC
            # subfolder = os.path.join(dataset, 'DFDC')
            for index, video_folder_name in enumerate(os.listdir(subfolder)):
                if index == opt.max_videos:
                    break
                if os.path.isdir(os.path.join(subfolder, video_folder_name)):
                    paths.append(os.path.join(subfolder, video_folder_name))

    # print('经过for dataset in sets后，此时paths=', paths)
    mgr = Manager()
    train_dataset = mgr.list()  # 此处train_dataset的值为   []
    validation_dataset = mgr.list()  # 此处train_dataset的值为 []

    '''
    绘制进度条，对paths中的元素执行read_frames函数
    train_dataset、validation_dataset是read_frames的两个参数
    '''
    with Pool(processes=10) as p:
        with tqdm(total=len(paths)) as pbar:
            for v in p.imap_unordered(
                    partial(read_frames, train_dataset=train_dataset, validation_dataset=validation_dataset), paths):
                pbar.update()

    train_samples = len(train_dataset)  # 长度
    train_dataset = shuffle_dataset(train_dataset)  # 此处输出的结果是一个包含标签的array数组

    validation_samples = len(validation_dataset)
    validation_dataset = shuffle_dataset(validation_dataset)  # 此处输出的结果是一个包含标签的array数组

    # Print some useful statistics
    print("Train images:", len(train_dataset), "Validation images:", len(validation_dataset))
    print("__TRAINING STATS__")
    train_counters = collections.Counter(image[1] for image in train_dataset)
    print(train_counters)

    class_weights = train_counters[0] / train_counters[1]
    print("Weights", class_weights)

    print("__VALIDATION STATS__")
    val_counters = collections.Counter(image[1] for image in validation_dataset)
    print(val_counters)
    print("___________________")

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]))

    # Create the data loaders
    validation_labels = np.asarray([row[1] for row in validation_dataset])
    labels = np.asarray([row[1] for row in train_dataset])

    '''
    np.asarray([row[0] for row in train_dataset])会报错
    因为图片的长宽没有统一，所以会出现ValueError: setting an array element with a sequence.（一维后形状不统一）
    解决办法：np.asarray中加入dtype=object
    image1 = [row[0] for row in train_dataset]
    print(np.asarray(image1,dtype=object))
    '''
    train_dataset = DeepFakesDataset(np.asarray([row[0] for row in train_dataset], dtype=object), labels,
                                     config['model']['image-size'])

    dl = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                     batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                     pin_memory=False, drop_last=False, timeout=0,
                                     worker_init_fn=None, prefetch_factor=2,
                                     persistent_workers=False)
    del train_dataset

    validation_dataset = DeepFakesDataset(np.asarray([row[0] for row in validation_dataset], dtype=object),
                                          validation_labels, config['model']['image-size'], mode='validation')
    val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=config['training']['bs'], shuffle=True,
                                         sampler=None,
                                         batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                         pin_memory=False, drop_last=False, timeout=0,
                                         worker_init_fn=None, prefetch_factor=2,
                                         persistent_workers=False)
    del validation_dataset

    model = model.cuda()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    counter = 0
    not_improved_loss = 0
    previous_loss = math.inf
    for t in range(starting_epoch, opt.num_epochs + 1):
        bar = ChargingBar('EPOCH #' + str(t), max=(len(dl) * config['training']['bs']) + len(val_dl))

        #
        # if not_improved_loss == opt.patience:
        #     break
        # counter = 0
        #
        # total_loss = 0
        # total_val_loss = 0
        #
        # bar = ChargingBar('EPOCH #' + str(t), max=(len(dl) * config['training']['bs']) + len(val_dl))
        # train_correct = 0
        # positive = 0
        # negative = 0
        # for index, (images, labels) in enumerate(dl):
        #     images = np.transpose(images, (0, 3, 1, 2))
        #     labels = labels.unsqueeze(1)
        #     images = images.cuda()
        #     # images = images.to(device)
        #
        #     y_pred = model(images)
        #     y_pred = y_pred.cpu()
        #     loss = loss_fn(y_pred, labels)
        #
        #     corrects, positive_class, negative_class = check_correct(y_pred, labels)
        #     train_correct += corrects
        #     positive += positive_class
        #     negative += negative_class
        #     optimizer.zero_grad()
        #
        #     loss.backward()
        #
        #     optimizer.step()
        #     counter += 1
        #     total_loss += round(loss.item(), 2)
        #     for i in range(config['training']['bs']):
        #         bar.next()
        #
        #     if index % 1200 == 0:
        #         print("\nLoss: ", total_loss / counter, "Accuracy: ",
        #               train_correct / (counter * config['training']['bs']), "训练为真数量（Train 0s）: ", negative,
        #               "训练为假数量（Train 1s）:", positive)

        train_correct = 0
        positive = 0
        negative = 0

        train_pre = []
        train_labels = []

        for index, (images, labels) in enumerate(dl):
            images = np.transpose(images, (0, 3, 1, 2))
            labels = labels.unsqueeze(1)
            images = images.cuda()
            # images = images.to(device)

            y_pred = model(images)
            y_pred = y_pred.cpu()

            train_pre.append(y_pred.detach().numpy())
            train_labels.append(labels.numpy())

            loss = loss_fn(y_pred, labels)

            corrects, positive_class, negative_class = check_correct(y_pred, labels)
            train_correct += corrects
            positive += positive_class
            negative += negative_class

            bar.next()

        train_pre = np.concatenate(train_pre)
        train_labels = np.concatenate(train_labels)
        auc_score = roc_auc_score(train_labels, train_pre)
        print("模型的训练集AUC Score:", auc_score)

        train_correct /= train_samples
        print("模型训练集准确率ACC:", train_correct)

        val_counter = 0
        val_correct = 0
        val_positive = 0
        val_negative = 0
        true_pred1num = 0
        true_pred0num = 0

        # 创建列表，计算模型AUC
        all_pre = []
        all_labels = []

        # train_correct /= train_samples
        #
        # total_loss /= counter
        for index, (val_images, val_labels) in enumerate(val_dl):
            val_images = np.transpose(val_images, (0, 3, 1, 2))

            val_images = val_images.cuda()
            # val_images = val_images.to(device)
            val_labels = val_labels.unsqueeze(1)

            val_pred = model(val_images)
            val_pred = val_pred.cpu()

            all_pre.append(val_pred.detach().numpy())
            all_labels.append(val_labels.numpy())

            val_loss = loss_fn(val_pred, val_labels)
            # total_val_loss += round(val_loss.item(), 2)
            corrects, positive_class, negative_class = check_correct(val_pred, val_labels)
            # corrects, positive_class, negative_class, true_pred1, true_pred0 = check_correct_01(val_pred, val_labels)
            val_correct += corrects
            val_positive += positive_class
            val_negative += negative_class
            val_counter += 1
            true_pred1num += 1
            true_pred0num += 1
            #             true_pred1num += true_pred1
            #             true_pred0num += true_pred0
            bar.next()

        scheduler.step()
        bar.finish()

        all_pre = np.concatenate(all_pre)
        all_labels = np.concatenate(all_labels)
        auc_score = roc_auc_score(all_labels, all_pre)
        print("模型的AUC Score:", auc_score)

        val_correct /= validation_samples
        print("模型准确率ACC:", val_correct)

        # total_val_loss /= val_counter
        # val_correct /= validation_samples
        # if previous_loss <= total_val_loss:
        #     print("Validation loss did not improved")
        #     not_improved_loss += 1
        # else:
        #     not_improved_loss = 0
        #
        # previous_loss = total_val_loss
        # print('当前的ACC，公式train_correct /= train_samples中公式train_correct指的是：', train_correct)
        # print('当前的ACC，公式train_correct /= train_samples中公式train_samples指的是：', train_samples)
        # print('结束后的真假比例：', negative, positive)
        # print("#" + str(t) + "/" + str(opt.num_epochs) + " loss:" +
        #       str(total_loss) + " accuracy:" + str(train_correct) + " val_loss:" + str(
        #     total_val_loss) + " val_accuracy:" + str(val_correct) + " val_0s:" + str(val_negative) + "/" + str(
        #     np.count_nonzero(validation_labels == 0)) + " val_1s:" + str(val_positive) + "/" + str(
        #     np.count_nonzero(validation_labels == 1)))
        # print("true_pred1num:", true_pred1num)
        # print("true_pred0num:", true_pred0num)

        #         if not os.path.exists(MODELS_PATH):
        #             os.makedirs(MODELS_PATH)
        #         torch.save(model.state_dict(),
        #                    os.path.join(MODELS_PATH, "efficientnet_checkpoint" + str(t) + "_" + opt.dataset))
        now = datetime.now()
        print("结束时间：", now.strftime('%Y-%m-%d %H:%M:%S'))