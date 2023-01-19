import torch
import random
import numpy as np
from itertools import cycle
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, f1_score, confusion_matrix


# 自定义的一些工具


def plot_confusion_matrix(y_true, y_pred, savename, title, classes):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    cm = confusion_matrix(y_true, y_pred)
    # 在混淆矩阵中每格的概率值
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        plt.text(x_val, y_val, "%0.4f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename)
    plt.close()
    # plt.show()


# def plot_curve(epoch_list, train_loss, train_acc, test_acc, out_dir: str):  # 损失 精度 画图功能函数
#     """
#      绘制训练和验证集的loss曲线/acc曲线
#      :param epoch_list: 迭代次数
#      :param train_loss: 训练损失
#      # :param test_acc:   训练测试精度
#      :param train_acc:  训练精度
#      :param out_dir:    保存路径
#      :return:
#      """
#     epoch = epoch_list
#     plt.subplot(2, 1, 1)
#     plt.plot(epoch, train_acc, label="train_acc")
#     plt.plot(epoch, test_acc, label="test_acc")
#
#     # plt.plot(epoch, test_acc, label="Test_acc")
#     plt.title('{}'.format(out_dir.split('\\')[-1].split('.')[0]))
#     plt.ylabel('accuracy')
#     plt.legend(loc='best')
#     plt.subplot(2, 1, 2)
#     plt.plot(epoch, train_loss, label="train_loss")
#     plt.xlabel('{}'.format(out_dir.split('\\')[-1].split('.')[0]))
#     plt.ylabel('loss')
#     plt.legend(loc='best')
#     plt.savefig(str(out_dir))
#     plt.close()
#     # plt.show()


def plot_curve(train_loss, savename, title):  # 画训练集损失
    plt.plot(train_loss, label="train_loss")
    plt.title('{}'.format(title))
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.savefig('{}.png'.format(savename))
    plt.close()


def plot_curve_acc(train_acc, savename, title):  # 画训练集损失
    plt.plot(train_acc, label="train_acc")
    plt.title('{}'.format(title))
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.savefig('{}.png'.format(savename))
    plt.close()


def acu_show(label, target, savename, title):
    fpr, tpr, thresholds = metrics.roc_curve(label, target)
    roc_auc = metrics.auc(fpr, tpr)  # auc为Roc曲线下的面积

    plt.plot(fpr, tpr, 'r', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate')  # 横坐标是fpr
    plt.ylabel('True Positive Rate')  # 纵坐标是tpr
    plt.title(title, fontsize=10)
    plt.savefig(savename)
    plt.close()
