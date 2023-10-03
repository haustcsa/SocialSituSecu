from Evaluate import *
import numpy as np

Invalid_Sign = -1

def costF(w,data_mat,g_truth,IsContinuous):
    K, L, M = data_mat.shape[0], data_mat.shape[1], data_mat.shape[2]
    totalcost = 0.0
    for k in range(K):
        for i in range(L):
            for m in range(M):
                if data_mat[k,i,m] != g_truth[i,m]:
                    if (data_mat[k,i,m] != Invalid_Sign) and (g_truth[i,m] != Invalid_Sign):
                        if IsContinuous[m]:
                            delta = data_mat[k,i,m] - g_truth[i,m]
                            totalcost += (w[k]*(delta*delta))
                        else:
                            totalcost += w[k]
    return totalcost

def Update_Weight(truth, data_mat,IsContinuous):
    K, L, M = data_mat.shape[0], data_mat.shape[1], data_mat.shape[2]
    num_Claims1 = np.zeros(shape=(K,), dtype=np.double)
    num_Claims2 = np.zeros(shape=(K,), dtype=np.double)
    score1 = 0.0
    score2 = 0.0
    for m in range(M):
        if IsContinuous[m]:
            delta = (truth[:,m] - data_mat[:,:,m]) * (data_mat[:,:,m] != Invalid_Sign).astype('int')
            score1 += np.sum(delta*delta)
            normize = np.sum(delta*delta, axis = 1)
            num_Claims1 += normize / max(normize)

        else:
            delta = (truth[:,m] != data_mat[:,:,m]).astype('int')
            delta = delta * (data_mat[:,:,m] != Invalid_Sign).astype('int')
            score2 += np.sum(delta)
            normize = np.sum(delta, axis = 1)
            if np.any(normize != 0):
                num_Claims2 += normize / max(normize)
    if sum(num_Claims1) != 0:
        num_Claims1 = num_Claims1 / sum(num_Claims1)
    if sum(num_Claims2) != 0:
        num_Claims2 = num_Claims2 / sum(num_Claims2)
    num_Claims = num_Claims1 + num_Claims2
    score = max(num_Claims)
    w_new = -np.log(num_Claims / score + 1e-300) + 0.00001
    return w_new

def getCount(data_mat, IsContinuous):
    valid_mat = (data_mat != Invalid_Sign).astype('int')
    Continuous_count = np.sum(valid_mat[:,:,IsContinuous], axis = (1,2))
    total_count = np.sum(valid_mat[:,:,:], axis = (1,2))
    cat_count = total_count - Continuous_count
    return Continuous_count, cat_count

def calcEffect(ground_truth,truth,weight,data_mat,valid_mat,IsContinuous):
    K, L, M = data_mat.shape[0], data_mat.shape[1], data_mat.shape[2]
    con_count, cat_count = getCount(data_mat, IsContinuous)
    avg_error = 0.0
    avg_MNAD = 0.0
    avg_MAD = 0.0
    avg_RMSE = 0.0
    ekg_list = []
    ekctd_list = []
    for k in range(K):
        total_error, MNAD, MAD, RMSE = evaluate(data_mat[k,:,:], ground_truth, valid_mat, IsContinuous)
        ekg_list.append((total_error, MNAD, MAD, RMSE))
        total_error_CTD, MNAD_CTD, MAD_CTD, RMSE_CTD = evaluate(data_mat[k,:,:], truth, valid_mat, IsContinuous)
        ekctd_list.append((total_error_CTD, MNAD_CTD, MAD_CTD, RMSE_CTD))
        total_error2 = np.abs(total_error_CTD - total_error)
        MNAD2 = np.abs(MNAD - MNAD_CTD)
        MAD2 = np.abs(MAD - MAD_CTD)
        RMSE2 = np.abs(RMSE - RMSE_CTD)
        avg_error += total_error2 * cat_count[k]
        avg_MNAD += MNAD2 * con_count[k]
        avg_MAD += MAD2 * con_count[k]
        avg_RMSE += RMSE2 * con_count[k]
    avg_error = avg_error / np.sum(cat_count)
    avg_MNAD = avg_MNAD / np.sum(con_count)
    avg_MAD = avg_MAD / np.sum(con_count)
    avg_RMSE = avg_RMSE / np.sum(con_count)
    return avg_error, avg_MNAD, avg_MAD, avg_RMSE, ekg_list, ekctd_list