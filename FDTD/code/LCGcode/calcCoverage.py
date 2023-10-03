import numpy as np

from LCGcode.DCTree import *


def cmp(a, op, b):
    if op == '=':
        return a == b
    elif op == '!=':
        return a != b
    elif op == '<':
        return a < b
    elif op == '>':
        return a > b
    elif op == '<=':
        return a <= b
    elif op == '>=':
        return a >= b
    assert False

def isTrue(dc,t1,t2,attList):
    for clause in dc.clause_list:
        if clause.single_entity:
            A = t1[attList.index(clause.left)]
            if clause.with_constant:
                B = clause.constant
            else:
                B = t1[attList.index(clause.right)]
            if not cmp(A,clause.op,B):
                return False
        else:
            A = t1[attList.index(clause.left)]
            B = t2[attList.index(clause.right)]
            if not cmp(A,clause.op,B):
                return False
    return True


def getSingleCoverage(trueVal,dc,attList):
    L, M = trueVal.shape[0], trueVal.shape[1]
    num = 0
    for i in range(L):
        if isTrue(dc, trueVal[i], None, attList):
            num += 1
    return num / (L)

def getNotSingleCoverageForce(trueVal,dc,attList):
    L, M = trueVal.shape[0],trueVal.shape[1]
    num = 0
    for i in range(L):
        for j in range(L):
            if i == j: continue
            if isTrue(dc,trueVal[i],trueVal[j],attList):
                num += 1
                print(i,j,trueVal[i],trueVal[j])
    return num / (L*(L-1))

def getNotSingleCoverage(trueVal,dc,attList):
    tree = DCTree()
    tree.add_DC(dc,0)

    data_mat = np.array([trueVal])

    worker = DCTreeWorker(tree,data_mat,attList)
    worker.work()

    num = worker.edgeNumber
    L, M = trueVal.shape[0],trueVal.shape[1]
    print(num)
    return num / (L*(L-1))

# 输入真值表、dcs、属性映射
# 返回一个长度为len(dcs)的list表示每个dc的coverage
# coverage：违反该dc的比例
def calcCoverage(trueVal,dcs,attList):
    L, M = trueVal.shape[0],trueVal.shape[1]
    dcList = list(dcs)
    ans = [0] * len(dcList)
    for dcId in range(len(dcList)):
        if dcList[dcId].Is_Single_Entity():
            ans[dcId] = getSingleCoverage(trueVal,dcList[dcId],attList)
        else:
            ans[dcId] = getNotSingleCoverage(trueVal, dcList[dcId], attList)
    return ans

if __name__ == '__main__':
    pass