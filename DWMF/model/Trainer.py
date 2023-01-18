from tkinter.tix import Tree
import torch.optim

from custom_tools import *
import time
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from network.model import BertClassModel, LsTModel, FusionModel, VitModel
from dataset import ClassDataset
import torch.nn as nn

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True


class Trainer(object):
    def __init__(self,
                 epochs,
                 learn_rate,
                 batch_size,
                 bert_path,
                 model_chose):

        super(Trainer, self).__init__()

        chose = ['FusionModel', 'BertModel', 'LstModel', 'VitModel']
        if model_chose not in chose:
            raise NameError("Model_combination should be one of {}, But you have chosen '{}', please correct it".
                            format(chose, model_chose))

        self.class_number = 2
        self.batch_size = batch_size
        self.clip = -1
        self.epochs = epochs
        self.bert_lr = learn_rate
        self.model_chose = model_chose

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.NLLLoss().to(self.device)

        if model_chose == 'FusionModel':
            model = FusionModel(bert_path)
            self.optimizer = self.get_optimizer(model)

        elif model_chose == 'BertModel':
            model = BertClassModel(bert_path)
            self.optimizer = self.get_optimizer(model)

        elif model_chose == 'LstModel':
            model = LsTModel()
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.bert_lr)

        elif model_chose == 'VitModel':
            model = VitModel()
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.bert_lr)

        self.model = model.to(self.device)

    def train(self, epoch, train_loader, train_len):

        self.model.train()
        correct = 0
        loss_list = []
        train_loss, train_acc, = 0, 0
        for batch_idx, batch in enumerate(train_loader):
            bert_ids, bert_mask, img_array, lstm_ids, target = batch[0], batch[1], batch[2], batch[3], batch[4]

            bert_ids = bert_ids.long().to(self.device)
            bert_mask = bert_mask.long().to(self.device)
            img_array = img_array.to(self.device)
            lstm_ids = lstm_ids.to(self.device)
            target = target.long().to(self.device)

            if self.model_chose == 'FusionModel':
                output = self.model(bert_ids, bert_mask, img_array, lstm_ids)
            elif self.model_chose == 'BertModel':
                output = self.model(bert_ids, bert_mask)
            elif self.model_chose == 'LstModel':
                output = self.model(lstm_ids)
            elif self.model_chose == 'VitModel':
                output = self.model(img_array)

            loss = self.criterion(output, target)

            self.optimizer.zero_grad()
            if self.clip > 0:  # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

            loss.backward()
            self.optimizer.step()
            train_loss += loss * target.size(0)
            argmax = torch.argmax(output, 1)
            train_acc += (argmax == target).sum()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            if batch_idx % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * self.batch_size,
                    train_len, 100. * batch_idx * self.batch_size / train_len,
                    loss.item()))

        train_loss = torch.true_divide(train_loss, train_len)
        train_acc = torch.true_divide(train_acc, train_len)
        print('Train set: Average loss: {:.6f}, Accuracy: {}/{} ({:.5f}%)'.format(
            train_loss, correct, train_len, 100. * correct / train_len))
        return train_loss, train_acc

    def evaluate(self, epoch, test_loader, test_len, model_chose):
        global best_acc
        correct_test, test_acc = 0, 0
        self.model.eval()
        with torch.no_grad():
            tar, argm = [], []
            for test_idx, test_batch in enumerate(test_loader):
                bert_ids, bert_mask, img_array, lstm_ids, target = \
                    test_batch[0], test_batch[1], test_batch[2], test_batch[3], test_batch[4]

                bert_ids = bert_ids.long().to(self.device)
                bert_mask = bert_mask.long().to(self.device)
                img_array = img_array.to(self.device)
                lstm_ids = lstm_ids.to(self.device)
                target = target.long().to(self.device)

                if self.model_chose == 'FusionModel':
                    output = self.model(bert_ids, bert_mask, img_array, lstm_ids)
                elif self.model_chose == 'BertModel':
                    output = self.model(bert_ids, bert_mask)
                elif self.model_chose == 'LstModel':
                    output = self.model(lstm_ids)
                elif self.model_chose == 'VitModel':
                    output = self.model(img_array)

                argmax = torch.argmax(output, 1)
                test_acc += (argmax == target).sum()
                pred_test = output.data.max(1, keepdim=True)[1]
                correct_test += pred_test.eq(target.data.view_as(pred_test)).cpu().sum()
                torch.cuda.empty_cache()

                tar.extend(target.cpu().numpy())
                argm.extend(argmax.cpu().numpy())

            test_acc = torch.true_divide(test_acc, test_len)

            print('\ntest set: Accuracy: {}/{} ({:.5f}%), Best_Accuracy({:.5f})'.format(
                correct_test, test_len, 100. * correct_test / test_len, best_acc))
            if test_acc > best_acc:
                best_acc = test_acc
                print('The effect becomes better and the parameters are saved .......')
                weight = r'result/{}.pt'.format(model_chose)
                torch.save(self.model.state_dict(), weight)  # 这里会存储迄今最优模型的参数

                p = precision_score(tar, argm, average='macro')
                recall = recall_score(tar, argm, average='macro')
                f1 = f1_score(tar, argm, average='macro')

                acu_show(label=argm, target=tar,
                         savename=r"result/Roc_Auc_{}".format(model_chose),
                         title='Roc_Auc_{}'.format(model_chose))

                plot_confusion_matrix(y_true=tar, y_pred=argm,
                                      savename=r"result/Confusion_Matrix_{}.png".format(model_chose),
                                      title=r"Confusion_Matrix_{}".format(model_chose),
                                      classes=['Normal', 'Malicious'])

                result_text = r'result/{}_text.txt'.format(model_chose)
                file_handle = open(result_text, mode='a+')
                file_handle.write('epoch:{},test_acc:{}, p:{}, recall:{},f1_score:{}\n'.format(
                    epoch, best_acc, p, recall, f1
                ))
                file_handle.close()
            return test_acc

    def get_optimizer(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        new_param = ['transitions', 'hidden2label.weight', 'hidden2label.bias']

        optimizer_grouped_parameters = [
            {'params':
                 [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and not
                 any(nd in n for nd in new_param)],
             'weight_decay': 1e-5
             },
            {'params':
                 [p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                  and not any(nd in n for nd in new_param)],
             'weight_decay': 0.0},
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.bert_lr, eps=1e-8)
        return optimizer


def mian(epochs, learn_rate, batch_size, bert_path, train_set, test_set, model_chose):
    print(f'choose model name {model_chose}')
    print(f'use gpu {torch.cuda.get_device_name()}')

    train_len, test_len = len(train_set), len(test_set)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    loss_train, acc_train, acc_test = [], [], []
    start = time.time()
    T = Trainer(epochs, learn_rate, batch_size, bert_path, model_chose)
    for epoch in range(epochs):
        train_loss, train_acc = T.train(epoch=epoch,
                                        train_loader=train_loader,
                                        train_len=train_len)

        test_acc = T.evaluate(epoch=epoch,
                              test_loader=test_loader,
                              test_len=test_len,
                              model_chose=model_chose)

        if torch.cuda.is_available():

            loss_train.append(train_loss.cuda().data.cpu().numpy())
            acc_train.append(train_acc.cuda().data.cpu().numpy())
        else:
            loss_train.append(train_loss.detach().numpy())
            acc_train.append(train_acc.detach().numpy())

        print("........................ Next ........................")

    end = time.time()
    train_time = end - start
    print("训练时间长度为  ==== > {} s".format(train_time))

    plot_curve(loss_train,
               savename=r"result/train_Loss_{}".format(model_chose),
               title=r"Train_Loss_{}".format(model_chose))  # 画出训练图像

    plot_curve_acc(acc_train,
                   savename=r"result/train_Acc_{}".format(model_chose),
                   title=r"Train_Acc_{}".format(model_chose))  # 画出训练图像


if __name__ == '__main__':
    best_acc = 0

    epochs_ = 10
    batch_size_ = 64
    learn_rate_ = 2e-5

    bert_path_ = r'bert-base-chinese'
    train_txt_ = r'data/train.csv'
    test_txt_ = r'data/test.csv'
    img_file_ = r'D:\train_img'

    train_set_ = ClassDataset(dataset=train_txt_,
                              img_file=img_file_,
                              state='train')

    test_set_ = ClassDataset(dataset=test_txt_,
                             img_file=img_file_,
                             state='test')

    # model_chose = ['FusionModel', 'BertModel', 'LstModel', 'VitModel']
    model_chose_ = 'LstModel'  # todo 改模型名称，上面四个选择

    mian(epochs=epochs_,
         learn_rate=learn_rate_,
         batch_size=batch_size_,
         bert_path=bert_path_,
         train_set=train_set_,
         test_set=test_set_,
         model_chose=model_chose_)
