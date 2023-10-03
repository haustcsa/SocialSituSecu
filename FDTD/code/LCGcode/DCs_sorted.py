import random
import numpy as np

class ClauseComparator:

    def __init__(self,dc_list,data_mat,Att_list,sampleTime=100):
        self.val = dict()
        self.frequency = dict()
        self.violate = dict()
        K, L, M = data_mat.shape[0], data_mat.shape[1], data_mat.shape[2]
        N = np.min([sampleTime,L])

        sampleId = random.sample(range(0,L),N)


        for i in range(len(dc_list)):
            dc = dc_list[i]
            if dc.Is_Single_Entity():
                continue
            for clause in dc.clause_list:
                if clause not in self.val:
                    self.val[clause] = 0
                    self.frequency[clause] = 0
                    self.violate[clause] = 0
                self.frequency[clause] += 1 # 出现次数
        for clause in self.val:
            for x in range(N):
                for y in range(N):
                    if x == y: continue
                    if clause.MayBeTrue(sampleId[x], sampleId[y], data_mat, Att_list):
                        self.violate[clause] += 1
            self.violate[clause] /= N*N-N # 成立的比例
            # 成立的比例越小、频率越大的越靠前
            self.val[clause] = self.violate[clause] / self.frequency[clause]

