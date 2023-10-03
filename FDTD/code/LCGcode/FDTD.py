from DCs_parser import *
import numpy as np
from LCG import *
from Evaluate import *
import pickle
import itertools
import time

Invalid_Sign = -1


def outputTruth(truth_val, decode_dict, IsContinuous):
    if decode_dict is None:
        return
    print('Truth:')
    L, M = truth_val.shape[0], truth_val.shape[1]
    for i in range(L):
        this_line = ''
        for m in range(M):
            if m > 0:
                this_line += ','
            if not IsContinuous[m]:
                this_line += decode_dict[int(round(truth_val[i, m]))]
            else:
                this_line += str(truth_val[i, m])
        print(this_line)


def Update_Weight(truth, data_mat, IsContinuous, w_dim=1):
    K, L, M = data_mat.shape[0], data_mat.shape[1], data_mat.shape[2]
    num_Claims1 = np.zeros(shape=(K,), dtype=np.double)
    num_Claims2 = np.zeros(shape=(K,), dtype=np.double)
    score1 = 0.0
    score2 = 0.0
    if w_dim == 2:
        w_new = np.zeros(shape=(K, M), dtype=np.double)
    for m in range(M):
        # if IsContinuous[m]:
        #     delta = (truth[:, m] - data_mat[:, :, m]) * (data_mat[:, :, m] != Invalid_Sign).astype('int')
        #     score1 += np.sum(delta * delta)
        #     normize = np.sum(delta * delta, axis=1)
        #     if w_dim == 1:
        #         num_Claims1 += normize / max(normize)
        #     else:
        #         assert w_dim == 2
        #         if sum(normize) != 0:
        #             normize = normize / sum(normize)
        #             score = max(normize)
        #             w_new[:, m] = -np.log(normize / score + 1e-300) + 0.00001
        # else:
        delta = (truth[:, m] != data_mat[:, :, m]).astype('int')
        delta = delta * (data_mat[:, :, m] != Invalid_Sign).astype('int')
        score2 += np.sum(delta)
        normize = np.sum(delta, axis=1)
        if w_dim == 1:
            if np.any(normize != 0):
                num_Claims2 += normize / max(normize)
        else:
            assert w_dim == 2
            if sum(normize) != 0:
                normize = normize / sum(normize)
                score = max(normize)
                w_new[:, m] = -np.log(normize / score + 1e-300) + 0.00001
    if w_dim == 2:
        return w_new
    if sum(num_Claims1) != 0:
        num_Claims1 = num_Claims1 / sum(num_Claims1)
    if sum(num_Claims2) != 0:
        num_Claims2 = num_Claims2 / sum(num_Claims2)
    num_Claims = num_Claims1 + num_Claims2
    score = max(num_Claims)
    w_new = -np.log(num_Claims / score + 1e-300) + 0.00001
    return w_new


def Updata_Truth_Discrete(i, m, w, data_mat, IsContinuous):
    K, L, M = data_mat.shape[0], data_mat.shape[1], data_mat.shape[2]
    e, a = i, m
    claim_list = data_mat[:, e, a]
    claim_species = np.unique(claim_list, return_index=False)
    wk = np.zeros(shape=(claim_species.shape[0],))
    for k in range(len(claim_species)):
        if w.ndim == 1:
            wk[k] = np.sum((claim_list == claim_species[k]).astype(int) * w)  # Change here if partial coverage
        else:
            assert w.ndim == 2
            wk[k] = np.sum((claim_list == claim_species[k]).astype(int) * w[:, m])
    new_claim_species = claim_species[claim_species != Invalid_Sign]
    new_wk = wk[claim_species != Invalid_Sign]
    if len(new_wk) == 0:
        return Invalid_Sign
    return new_claim_species[np.argmax(new_wk)]


def Updata_Truth_Discrete_new(i, m, w, data_mat, IsContinuous):
    value_list = data_mat[:, i, m]
    value_dict = dict()
    for i in range(len(value_list)):
        if value_list[i] not in value_dict:
            value_dict[value_list[i]] = 0.0
        if w.ndim == 1:
            value_dict[value_list[i]] += w[i]
        else:
            assert w.ndim == 2
            value_dict[value_list[i]] += w[i, m]
    max_weight = 0.0
    best_ans = 0.0
    for value, weight in value_dict.items():
        if value == Invalid_Sign:
            continue
        if weight > max_weight:
            max_weight = weight
            best_ans = value
    if max_weight == 0.0:
        return Invalid_Sign
    return best_ans


def Updata_Truth(i, m, w, data_mat, IsContinuous):
    # if not IsContinuous[m]:
    return Updata_Truth_Discrete_new(i, m, w, data_mat, IsContinuous)
    # if np.all(data_mat[:, i, m] == Invalid_Sign):
    #     return Invalid_Sign
    # if w.ndim == 1:
    #     return np.average(data_mat[:, i, m][data_mat[:, i, m] != Invalid_Sign],
    #                       weights=w[data_mat[:, i, m] != Invalid_Sign])
    # else:
    #     assert w.ndim == 2
    #     return np.average(data_mat[:, i, m][data_mat[:, i, m] != Invalid_Sign],
    #                       weights=w[data_mat[:, i, m] != Invalid_Sign, m])


def FDTD(g_truth, data_mat, source_weight, dcs, Att_list, IsContinuous, save_block_To_file=False,
        load_block_From_file=False, datasetname='', only_generate_conflict_blocks=False, max_itor=10, decode_dict=None,
        run_mode=0):
    K, L, M = data_mat.shape[0], data_mat.shape[1], data_mat.shape[2]
    truth_mat = np.ones(shape=(L, M), dtype=np.double) * Invalid_Sign
    w = source_weight.copy()
    Cells_list, LCset_list, Partition_time, generate_time = LCG(data_mat, w, dcs, Att_list, IsContinuous,
                                                                save_block_To_file=save_block_To_file,
                                                                load_block_From_file=load_block_From_file,
                                                                datasetname=datasetname,
                                                                only_generate_conflict_blocks=only_generate_conflict_blocks)
    Annealing_time = 0
    starttime = time.clock()
    for itor in range(max_itor):
        print('CTD itor:' + str(itor))
        for i in range(L):
            for m in range(M):
                truth_mat[i, m] = Updata_Truth(i, m, w, data_mat, IsContinuous)
        # Update truth in each block
        for i in range(len(Cells_list)):
            if (run_mode != 2) and (run_mode != 3):
                Solve_Conflict_On_Continuous_Data(data_mat, w, truth_mat, Cells_list[i], LCset_list[i], IsContinuous)
            if (run_mode != 1) and (run_mode != 3):
                Solve_Conflict_On_Categorical_Data_with_Simulated_Annealing(g_truth, data_mat, w, truth_mat,
                                                                            Cells_list[i], LCset_list[i], IsContinuous)
        # Update weight
        w = Update_Weight(truth_mat, data_mat, IsContinuous, w_dim=source_weight.ndim)
    endtime = time.clock()
    return truth_mat, Partition_time, generate_time, endtime - starttime
