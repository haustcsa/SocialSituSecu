from __future__ import division

import math
import random
import logging
from pprint import pprint
Lables = {0: '1',
          1: '2',
          2: '3'}
random.seed(0)
def rand(a, b):
    return (b - a) * random.random() + a
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill] * J)
    return m
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))
def dsigmoid(x):
    return x * (1 - x)
def readfile():
    path = ''
    outnum = []
    with open(path, 'r', encoding="utf-8") as f:
        line = f.readline().split('\t')
        for i in range(len(line)):
            line_num = round(float(line[i]), 3)
            outnum.append(line_num)
    return outnum
def calculate(ni, nh, no):
    outnum = readfile()
    in_hidden = []
    hi_output = []
    t = 0
    for i in range(ni + 1):
        in_hidden.append([])
        for j in range(nh):
            in_hidden[i].append(outnum[t])
            t = t + 1
    for j in range(nh + 1):
        hi_output.append([])
        for k in range(no):
            hi_output[j].append(outnum[t])
            t = t + 1
    return in_hidden, hi_output
class NN:
    """ 三层反向传播神经网络 """
    def __init__(self, ni, nh, no):
        self.ni = ni
        self.nh = nh
        self.no = no
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        in_hidden, hi_output = calculate(a1, a2, a3)
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = in_hidden[i][j]
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = hi_output[j][k]
        self.bi = makeMatrix(1, self.nh)
        self.bo = makeMatrix(1, self.no)
        for j in range(self.nh):
            self.bi[0][j] = in_hidden[self.ni][j]
        for k in range(self.no):
            self.bo[0][k] = hi_output[self.nh][k]
    def update(self, inputs):
        if len(inputs) != self.ni :
            print('----', len(inputs))
            raise ValueError('与输入层节点数不符！')
        for i in range(self.ni):
            self.ai[i] = inputs[i]
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum - self.bi[0][j])
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum - self.bo[0][k])
        return self.ao[:]
    def backPropagate(self, targets, lr):
        """ 反向传播 """
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] = self.wo[j][k] + lr * change
        for k in range(self.no):
            change = output_deltas[k] * lr
            self.bo[0][k] = -change + self.bo[0][k]
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] + lr * change
        for j in range(self.nh):
            change = hidden_deltas[j] * lr
            self.bi[0][j] = -change + self.bi[0][j]
        error = 0.0
        error = 0.5 * ((targets[k] - self.ao[k]) ** 2) + error
        return error
    def test(self, patterns):
        count = 0
        for p in patterns:
            target = Lables[(p[1].index(1))]
            result = self.update(p[0])
            index = result.index(max(result))
             print(target, ' ', Lables[index])
            count += (target == Lables[index])
        accuracy = float(count / len(patterns))
        path = ''
        with open(path, 'w', encoding='utf-8') as f:
            for p in patterns:
                target = flowerLables[(p[1].index(1))]
                result = self.update(p[0])
                index = result.index(max(result))
                f.write(target + ' ' + Lables[index])
                f.write('\n')
    def weights(self):
        print('输入层权重:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('输出层权重:')
        for j in range(self.nh):
            print(self.wo[j])
    def train(self, patterns, iterations=a4, lr=a5):
        for i in range(iterations):
            error1 = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error1 = error1 + self.backPropagate(targets, lr)
                logging.info(self.backPropagate(targets, lr))
            if i % 100 == 0:
                print('error: %-.9f' % error1)
def loaddataset(filename):
    fp = open(filename)
    dataset = []
    labelset = []
    for i in fp.readlines():
        a = i.strip().split()
        dataset.append([float(j) for j in a[:len(a) - 1]])
        labelset.append(int(float(a[-1])))
    return dataset, labelset
def ir():
    dataset, labelset = loaddataset('')
    data = []
    for i in range(len(dataset)):
        ele = []
        ele.append(dataset[i])
        if labelset[i] == 1:
            ele.append([1, 0, 0])
        elif labelset[i] == 2:
            ele.append([0, 1, 0])
        else:
            ele.append([0, 0, 1])
        data.append(ele)
     index = [i for i in range(len(data))]
    random.shuffle(index)
    data_list = []
    for i in range(b1, b2):
        data_list.append(data[index[i]])
    for i in range(b2, b3):
        data_list.append(data[index[i]])
    trainDataSet = random.sample(data, b1)
    for testData in trainDataSet:
        data.remove(testData)
    testDataSet = random.sample(data, b4)
    training = trainDataSet[:]
    test = testDataSet[:]
    nn = NN(a1,a2,a3)
    nn.train(training, iterations=b5)
    nn.test(test)
if __name__ == '__main__':
    ir()