from ClusterPairs import *
import numpy as np
Invalid_Sign = -1

class PartitionRefiner:

    # 输入一个cluster pair和clause，和数据集
    # 返回从其refine出的cluster pairs的list
    @staticmethod
    def refine(clusterPair, clause, data_mat, Att_list):
        list = []
        if clause.single_entity:
            if clause.with_constant: # t1[a] op const
                list = PartitionRefiner.singleConstRefine(clusterPair, clause, data_mat, Att_list)
            else: # t1[a] op t1[b]
                list = PartitionRefiner.singleNoConstRefine(clusterPair, clause, data_mat, Att_list)
        else: # t1[a] op t2[b]
            if clause.op == '=':
                list = PartitionRefiner.eqJoin(clusterPair, clause, data_mat, Att_list)
            elif clause.op == '!=':
                list = PartitionRefiner.antiJoin(clusterPair, clause, data_mat, Att_list)
            else:
                list = PartitionRefiner.ineqJoin(clusterPair, clause, data_mat, Att_list,clause.op)
        return list

    @staticmethod
    def eqJoin(clusterPair, clause, data_mat, Att_list): # =
        assert clause.left in Att_list
        assert clause.right in Att_list
        att1 = Att_list.index(clause.left)
        att2 = Att_list.index(clause.right)
        # print("att : ", att1,att2)
        lhash,rhash = dict(),dict()

        K, L, M = data_mat.shape[0], data_mat.shape[1], data_mat.shape[2]
        # K个数据源，L个数据，M个属性

        for x in clusterPair.a:
            data = data_mat[:,x,att1]
            data = data[data != Invalid_Sign]
            for key in data:
                if key not in lhash:
                    lhash[key] = set()
                lhash[key].add(x)

        for y in clusterPair.b:
            data = data_mat[:, y, att2]
            data = data[data != Invalid_Sign]
            for key in data:
                if key not in rhash:
                    rhash[key] = set()
                rhash[key].add(y)

        list = []
        for key in lhash:
            if key not in rhash: continue
            cp = ClusterPair(lhash[key],rhash[key])
            list.append(cp)

            # for x in cp.a:
            #     for y in cp.b:
            #         print((data_mat[0][x][att1],data_mat[1][x][att1]),(data_mat[0][y][att2],data_mat[1][y][att2]))

        # list = []
        # for x in clusterPair.a:
        #     for y in clusterPair.b:
        #         flag = False
        #         for i in range(K):
        #             for j in range(K):
        #                 if data_mat[i][x][att1] == data_mat[j][y][att2]:
        #                     flag = True
        #         if flag:
        #             list.append(ClusterPair({x},{y}))
        return list

    @staticmethod
    def antiJoin(clusterPair, clause, data_mat, Att_list):  # !=
        assert clause.left in Att_list
        assert clause.right in Att_list
        att1 = Att_list.index(clause.left)
        att2 = Att_list.index(clause.right)
        lhash,rhash = dict(),dict()

        K, L, M = data_mat.shape[0], data_mat.shape[1], data_mat.shape[2]
        # K个数据源，L个数据，M个属性

        for x in clusterPair.a:
            data = data_mat[:,x,att1]
            data = data[data != Invalid_Sign]
            for key in data:
                if key not in lhash:
                    lhash[key] = set()
                lhash[key].add(x) # 可以取到这个点的点集

        for y in clusterPair.b:
            data = data_mat[:, y, att2]
            data = data[data != Invalid_Sign]
            if len(data) == 0: continue
            key = data[0]
            flag = True
            for key1 in data:
                if key != key1:
                    flag = False
                    break
            if flag:
                if key not in rhash:
                    rhash[key] = set()
                rhash[key].add(y)  # 只会取到这个点的点集

        # print(len(lhash),len(rhash))
        list = []
        lsum = set()
        for key in lhash:
            if key in rhash:
                cp = ClusterPair(lhash[key], clusterPair.b - rhash[key])
                list.append(cp)
            else:
                lsum.update(lhash[key])

        if len(lsum) > 0:
            cp = ClusterPair(lsum, clusterPair.b)
            list.append(cp)

        # list = []
        # for x in clusterPair.a:
        #     for y in clusterPair.b:
        #         flag = False
        #         for i in range(K):
        #             for j in range(K):
        #                 if data_mat[i][x][att1] != data_mat[j][y][att2]:
        #                     flag = True
        #         if flag:
        #             list.append(ClusterPair({x},{y}))

        return list

    @staticmethod
    def ineqJoin(clusterPair, clause, data_mat, Att_list, op):  # !=
        assert clause.left in Att_list
        assert clause.right in Att_list
        att1 = Att_list.index(clause.left)
        att2 = Att_list.index(clause.right)

        vl, vr = [], []
        K, L, M = data_mat.shape[0], data_mat.shape[1], data_mat.shape[2]
        # K个数据源，L个数据，M个属性

        if op == '<' or op == '<=':
            for x in clusterPair.a:
                data = data_mat[:, x, att1]
                data = data[data != Invalid_Sign]
                if len(data) == 0: continue
                v = min(data)
                vl.append((v, x))

            for y in clusterPair.b:
                data = data_mat[:, y, att2]
                data = data[data != Invalid_Sign]
                if len(data) == 0: continue
                v = max(data)
                vr.append((v, y))
            vl = sorted(vl) # (key,id)
            vr = sorted(vr)
        else:
            for x in clusterPair.a:
                data = data_mat[:, x, att1]
                data = data[data != Invalid_Sign]
                if len(data) == 0: continue
                v = max(data)
                vl.append((v, x))
            for y in clusterPair.b:
                data = data_mat[:, y, att2]
                data = data[data != Invalid_Sign]
                if len(data) == 0: continue
                v = min(data)
                vr.append((v, y))
            vl = sorted(vl,reverse=True)
            vr = sorted(vr,reverse=True)

        list = []
        allset = set((vr[i][1]) for i in range(len(vr)))
        i = 0
        for j in range(len(vr)):
            nowset = set()
            while i < len(vl) and ( (op == '<' and vl[i][0] < vr[j][0]) or
                                    (op == '<=' and vl[i][0] <= vr[j][0]) or
                                    (op == '>' and vl[i][0] > vr[j][0]) or
                                    (op == '>=' and vl[i][0] >= vr[j][0])):
                nowset.add(vl[i][1])
                i += 1
            if len(nowset) > 0:
                list.append(ClusterPair(nowset, allset))
            allset.remove(vr[j][1])
        return list

    @staticmethod
    def cmp(a,op,b):
        if a == Invalid_Sign or b == Invalid_Sign:
            return True
        if op == '=':
            return a == b
        elif op == '!=':
            return a != b
        elif op == '<':
            return a < b
        elif op == '>':
            return a > b
        elif op == '>=':
            return a >= b
        elif op == '<=':
            return a <= b
        assert False

    @staticmethod
    def singleConstRefine(clusterPair, clause, data_mat, Att_list):
        assert clause.left in Att_list
        att1 = Att_list.index(clause.left)
        K, L, M = data_mat.shape[0], data_mat.shape[1], data_mat.shape[2]
        # K个数据源，L个数据，M个属性

        cp = ClusterPair(clusterPair.a,clusterPair.b)
        for x in clusterPair.a:
            data = data_mat[:, x, att1]
            data = data[data != Invalid_Sign]
            flag = False
            for val in data:
                if PartitionRefiner.cmp(val, clause.op, clause.right):
                    flag = True
            if flag: # 如果存在一个取值会成立
               pass
            else:
                cp.a.remove(x)

        return [cp]

    @staticmethod
    def singleNoConstRefine(clusterPair, clause, data_mat, Att_list):
        assert clause.left in Att_list
        assert clause.right in Att_list
        att1 = Att_list.index(clause.left)
        att2 = Att_list.index(clause.right)
        K, L, M = data_mat.shape[0], data_mat.shape[1], data_mat.shape[2]
        # K个数据源，L个数据，M个属性

        cp = ClusterPair(clusterPair.a, clusterPair.b)
        for x in clusterPair.a:
            data1 = data_mat[:, x, att1]
            data1 = data1[data1 != Invalid_Sign]

            data2 = data_mat[:, x, att2]
            data2 = data2[data2 != Invalid_Sign]

            flag = False
            if clause.op == '=':
                if len(set(data1).intersection(set(data2))) > 0: # 存在两个相等的
                    flag = True
            elif clause.op == '!=':
                # 不成立当且仅当两个list的值完全一样
                if len(set(data1)) <= 1 and len(set(data2)) <= 1 \
                    and  len(set(data1).intersection(set(data2))) <= 1:
                    pass
                else:
                    flag = True
            elif clause.op == '<' or clause.op == '<=':
                if PartitionRefiner.cmp(np.min(data1), clause.op, np.max(data2)):
                    flag = True
            else:
                if PartitionRefiner.cmp(np.max(data1), clause.op, np.min(data2)):
                    flag = True
            if flag:  # 如果存在一个取值会成立
                pass
            else:
                cp.a.remove(x)

        return [cp]

